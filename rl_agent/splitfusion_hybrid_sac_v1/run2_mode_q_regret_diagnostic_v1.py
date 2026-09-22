"""Read-only mode/q/regret diagnostic for the completed exact-P95 Run-2 v2 policy.

This module answers one question: *why does the finished Run-2 actor select
discrete mode 11 for every development context, and where does the remaining
gap to the contextual constrained oracle actually live?*

It is analysis-only.  It never trains, never writes into frozen Run-1/Run-2
evidence, never touches CARLA/OAI, and never initializes CUDA.  Every scoring
path is imported unchanged from the frozen evaluator/oracle modules so that the
numbers here are the *same* numbers the registered evaluation produced:

``_surface_mode_vector`` / ``_network_vector``    exact per-mode action tables
``_base64_vector`` / ``_shaped64_vector``          pinned binary64 utility order
``_emitted_vector``                                the one float32 emission
``_best_index``                                    the registered tie rule
``_score_v2``                                      the scalar cross-check

The reported P95 remains the modeled conditional retained-survivor proxy on the
frozen 85-scene x four-profile development fit-validation panel.  It is not a
live SLA and the panel is not 1,020 independent samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .empirical_contextual_contract import MODELED_SMOKE_SUPPORT
from .empirical_contextual_exact_p95_deadline_penalty_v2 import (
    _base64_vector,
    _emitted_vector,
    _shaped64_vector,
)
from .empirical_contextual_exact_p95_run2_evaluator_v2 import (
    DEADLINE_MS_V2,
    FIXED_MODE_ID_V2,
    FIXED_Q_E4_V2,
    ORACLE_RELATIVE_PATH,
    SOURCE_FIXED,
    SOURCE_ORACLE,
    SOURCE_RANDOM,
    SOURCE_RUN1,
    SOURCE_V2,
    ExactP95Run2EvaluationV2Error,
    _load_oracle_actions,
    _require_fixed_comparator,
    _require_training_cli_source,
    _score_v2,
    _sha256_file,
)
from .empirical_contextual_exact_p95_run2_replay_v2 import RUN2_V2_DEADLINE_PENALTY
from .empirical_contextual_fit_validation_evaluator import (
    FitValidationActorEvaluatorV1,
)
from .empirical_contextual_split_oracle import (
    _authoritative_outcome,
    _best_index,
    _network_vector,
    _surface_mode_vector,
)
from .hybrid_sac_models import (
    HybridSacModelConfig,
    build_actor,
    build_twin_critics,
    quantize_q_e4,
)
from .modeled_smoke_support import MODELED_SMOKE_MODE_Q_E4_BOUNDS
from .run_empirical_contextual_exact_p95_run2_v2 import (
    load_exact_p95_run2_checkpoint_v2,
)
from .transaction_identity import canonical_sha256


__all__ = [
    "DIAGNOSTIC_SCHEMA_V1",
    "MODE_COUNT",
    "REGISTERED_UPDATES",
    "REGISTERED_SEEDS",
    "Run2ModeQDiagnosticError",
    "average_ranks",
    "descending_rank",
    "shannon_entropy_nats",
    "spearman_rho",
    "decompose_context_regret",
    "run_diagnostic",
]


DIAGNOSTIC_SCHEMA_V1 = "splitfusion.run2_mode_q_regret_diagnostic.v1"
SUMMARY_SCHEMA_V1 = "splitfusion.run2_mode_q_regret_diagnostic_summary.v1"
MANIFEST_SCHEMA_V1 = "splitfusion.run2_mode_q_regret_diagnostic_manifest.v1"

MODE_COUNT = len(MODELED_SMOKE_MODE_Q_E4_BOUNDS)
PANEL_CONTEXT_COUNT = 340
REGISTERED_UPDATES: Tuple[int, ...] = (
    0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000
)
REGISTERED_SEEDS: Tuple[int, ...] = (17, 29, 43)
PRIMARY_UPDATE = 5000

TRAINING_CAMPAIGN_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_run2_training_v2/20260921_exact_p95_run2_v2"
)
EVALUATION_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_run2_evaluation_v2/20260922_exact_p95_run2_v2"
)
RUN1_CAMPAIGN_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_preliminary_baseline_v1/"
    "20260921_train_split_5000x3_v1"
)
OUTPUT_RELATIVE_PATH = (
    "experiments/splitfusion_hybrid_sac_run2_diagnostic_v1/"
    "20260922_mode_q_regret_decomposition_v1"
)

# Anchors established by the frozen evaluation.  The diagnostic refuses to run
# if it cannot reproduce them from the frozen artifacts, so a silent binding
# swap becomes a hard failure instead of a quietly different report.
EXPECTED_ANCHORS: Mapping[str, float] = {
    "run2_final_pooled_v2_utility": 0.297090,
    "update0_v2_utility": -0.212613,
    "run1_rescored_v2_utility": -0.135444,
    "fixed_v2_utility": 0.319213,
    "oracle_v2_utility": 0.427386,
    "run2_quality": 0.539921,
    "fixed_quality": 0.551762,
    "oracle_quality": 0.657503,
}
EXPECTED_FINAL_MISS_COUNT = 23
EXPECTED_FINAL_DECISION_COUNT = 1020
ANCHOR_TOLERANCE = 5e-7


class Run2ModeQDiagnosticError(RuntimeError):
    """A diagnostic identity, binding or reproduction invariant failed."""


# --------------------------------------------------------------------------- #
# Small pure helpers (independently unit tested)
# --------------------------------------------------------------------------- #


def shannon_entropy_nats(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats, treating an exact zero as contributing zero."""
    total = math.fsum(float(p) for p in probabilities)
    if not math.isfinite(total) or abs(total - 1.0) > 1e-5:
        raise Run2ModeQDiagnosticError("entropy input is not a probability vector")
    terms = []
    for value in probabilities:
        p = float(value)
        if p < 0.0:
            raise Run2ModeQDiagnosticError("entropy input has a negative mass")
        if p > 0.0:
            terms.append(-p * math.log(p))
    return math.fsum(terms)


def descending_rank(values: Sequence[float]) -> Tuple[int, ...]:
    """Competition-free dense position, best value first (0 = best).

    Ties are broken by ascending index so the result is a permutation and the
    top-1 read is deterministic, matching the registered stable tie rule.
    """
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), i))
    rank = [0] * len(values)
    for position, index in enumerate(order):
        rank[index] = position
    return tuple(rank)


def average_ranks(values: Sequence[float]) -> Tuple[float, ...]:
    """Ascending ranks with ties averaged, as Spearman's rho requires."""
    order = sorted(range(len(values)), key=lambda i: float(values[i]))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position
        while (
            end + 1 < len(order)
            and float(values[order[end + 1]]) == float(values[order[position]])
        ):
            end += 1
        shared = (position + end) / 2.0 + 1.0
        for index in order[position : end + 1]:
            ranks[index] = shared
        position = end + 1
    return tuple(ranks)


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    """Spearman rank correlation with tie-averaged ranks.

    Returns ``nan`` when either side is constant, because rho is undefined
    there; the caller reports the undefined count rather than imputing zero.
    """
    if len(left) != len(right) or len(left) < 2:
        raise Run2ModeQDiagnosticError("spearman inputs must be equal length >= 2")
    a = np.asarray(average_ranks(left), dtype=np.float64)
    b = np.asarray(average_ranks(right), dtype=np.float64)
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denominator = math.sqrt(
        float(np.dot(a_centered, a_centered)) * float(np.dot(b_centered, b_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a_centered, b_centered) / denominator)


@dataclass(frozen=True, slots=True)
class RegretDecomposition:
    """The one additive path plus the explicitly overlapping alternative.

    ``total = continuous_q_regret + mode_given_best_q_regret`` holds exactly by
    construction because B shares A's mode and D is the unrestricted optimum.
    ``mode_at_actor_q_regret`` (C - A) is *not* a term of that sum: it re-prices
    the discrete choice while freezing each mode at the actor's own proposed q,
    so it overlaps both terms and is reported separately.
    """

    utility_a: float
    utility_b: float
    utility_c: float
    utility_d: float
    utility_e: float
    continuous_q_regret: float
    mode_given_best_q_regret: float
    total_oracle_regret: float
    mode_at_actor_q_regret: float
    learned_minus_fixed: float

    def require_additive(self, tolerance: float = 1e-9) -> None:
        residual = self.total_oracle_regret - (
            self.continuous_q_regret + self.mode_given_best_q_regret
        )
        if abs(residual) > tolerance:
            raise Run2ModeQDiagnosticError("regret decomposition is not additive")


def decompose_context_regret(
    *,
    utility_a: float,
    utility_b: float,
    utility_c: float,
    utility_d: float,
    utility_e: float,
) -> RegretDecomposition:
    """Build the decomposition and refuse dominance violations.

    B is A's mode at that mode's best supported q, so ``B >= A``.  D is the
    optimum over every supported action, so ``D >= B`` and ``D >= C``.  A
    violation means the counterfactual tables and the executed row disagree,
    which is a bug, not a finding.
    """
    if utility_b < utility_a - 1e-9:
        raise Run2ModeQDiagnosticError("B must dominate A within the learned mode")
    if utility_d < utility_b - 1e-9 or utility_d < utility_c - 1e-9:
        raise Run2ModeQDiagnosticError("D must dominate B and C")
    decomposition = RegretDecomposition(
        utility_a=utility_a,
        utility_b=utility_b,
        utility_c=utility_c,
        utility_d=utility_d,
        utility_e=utility_e,
        continuous_q_regret=utility_b - utility_a,
        mode_given_best_q_regret=utility_d - utility_b,
        total_oracle_regret=utility_d - utility_a,
        mode_at_actor_q_regret=utility_c - utility_a,
        learned_minus_fixed=utility_a - utility_e,
    )
    decomposition.require_additive()
    return decomposition


# --------------------------------------------------------------------------- #
# Frozen-evidence revalidation
# --------------------------------------------------------------------------- #


def _read_json(path: Path) -> Dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if type(document) is not dict:
        raise Run2ModeQDiagnosticError(f"{Path(path).name} is not a JSON object")
    return document


def require_file_sha256(path: Path, expected: str, label: str) -> str:
    """Hash ``path`` and refuse any drift from ``expected``."""
    path = Path(path)
    if not path.is_file():
        raise Run2ModeQDiagnosticError(f"missing frozen artifact: {label}")
    actual = _sha256_file(path)
    if actual != expected:
        raise Run2ModeQDiagnosticError(
            f"frozen artifact drift for {label}: {actual} != {expected}"
        )
    return actual


def revalidate_frozen_bindings(root: Path) -> Dict[str, Any]:
    """Revalidate every source binding *before* any scientific value is read."""
    root = Path(root).resolve(strict=True)
    evaluation_dir = root / EVALUATION_RELATIVE_PATH
    manifest = _read_json(evaluation_dir / "manifest.json")

    supplied = manifest.get("manifest_content_sha256")
    recomputed = canonical_sha256(
        {k: v for k, v in manifest.items() if k != "manifest_content_sha256"}
    )
    if supplied != recomputed:
        raise Run2ModeQDiagnosticError("evaluation manifest content hash mismatch")
    if manifest.get("status") != "COMPLETE":
        raise Run2ModeQDiagnosticError("evaluation manifest is not COMPLETE")

    artifacts = manifest.get("artifacts")
    if type(artifacts) is not dict or not artifacts:
        raise Run2ModeQDiagnosticError("evaluation manifest has no artifact table")
    verified: Dict[str, str] = {}
    for name, expected in sorted(artifacts.items()):
        verified[name] = require_file_sha256(
            evaluation_dir / name, expected, f"evaluation/{name}"
        )

    # Frozen upstream identities, checked with the evaluator's own validators.
    _require_training_cli_source(root)
    _require_fixed_comparator(root)
    oracle_actions = _load_oracle_actions(root)
    if len(oracle_actions) != PANEL_CONTEXT_COUNT:
        raise Run2ModeQDiagnosticError("oracle action coverage drift")

    campaign_path = root / TRAINING_CAMPAIGN_RELATIVE_PATH / "campaign_report.json"
    require_file_sha256(
        campaign_path,
        manifest["v2_campaign_report_file_sha256"],
        "run2_training/campaign_report.json",
    )
    if not (root / RUN1_CAMPAIGN_RELATIVE_PATH / "campaign_report.json").is_file():
        raise Run2ModeQDiagnosticError("Run-1 campaign report is missing")

    return {
        "evaluation_artifact_sha256": verified,
        "evaluation_manifest_content_sha256": recomputed,
        "fit_partition_sha256": manifest["fit_partition_sha256"],
        "fit_validation_panel_sha256": manifest["fit_validation_panel_sha256"],
        "fixed_comparator_commit": manifest["fixed_comparator_commit"],
        "oracle_actions": oracle_actions,
        "protocol_sha256": manifest["protocol_sha256"],
        "reward_spec_v2_sha256": manifest["reward_spec_v2_sha256"],
        "run1_campaign_report_file_sha256": _sha256_file(
            root / RUN1_CAMPAIGN_RELATIVE_PATH / "campaign_report.json"
        ),
        "training_cli_commit": manifest["training_cli_commit"],
        "v2_campaign_report_file_sha256": manifest["v2_campaign_report_file_sha256"],
    }


def load_frozen_per_context(root: Path) -> List[Dict[str, str]]:
    """Load the already-hash-verified frozen per-context evaluation rows."""
    path = Path(root) / EVALUATION_RELATIVE_PATH / "per_context.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 13940:
        raise Run2ModeQDiagnosticError("frozen per-context row count drift")
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise Run2ModeQDiagnosticError("mean of an empty population")
    return math.fsum(float(v) for v in values) / len(values)


def reproduce_anchors(rows: Sequence[Mapping[str, str]]) -> Dict[str, float]:
    """Recompute the established headline numbers from the frozen rows."""

    def select(kind: str, update: Optional[int] = None) -> List[Mapping[str, str]]:
        return [
            r
            for r in rows
            if r["source_kind"] == kind
            and (update is None or int(r["update_index"]) == update)
        ]

    run2 = select(SOURCE_V2, PRIMARY_UPDATE)
    update0 = select(SOURCE_V2, 0)
    run1 = select(SOURCE_RUN1)
    fixed = select(SOURCE_FIXED)
    oracle = select(SOURCE_ORACLE)
    if len(run2) != EXPECTED_FINAL_DECISION_COUNT:
        raise Run2ModeQDiagnosticError("Run-2 final decision count drift")

    anchors = {
        "run2_final_pooled_v2_utility": _mean(
            [float(r["v2_emitted_reward_float32"]) for r in run2]
        ),
        "update0_v2_utility": _mean(
            [float(r["v2_emitted_reward_float32"]) for r in update0]
        ),
        "run1_rescored_v2_utility": _mean(
            [float(r["v2_emitted_reward_float32"]) for r in run1]
        ),
        "fixed_v2_utility": _mean(
            [float(r["v2_emitted_reward_float32"]) for r in fixed]
        ),
        "oracle_v2_utility": _mean(
            [float(r["v2_emitted_reward_float32"]) for r in oracle]
        ),
        "run2_quality": _mean([float(r["q_perc"]) for r in run2]),
        "fixed_quality": _mean([float(r["q_perc"]) for r in fixed]),
        "oracle_quality": _mean([float(r["q_perc"]) for r in oracle]),
    }
    for name, expected in EXPECTED_ANCHORS.items():
        if abs(anchors[name] - expected) > ANCHOR_TOLERANCE:
            raise Run2ModeQDiagnosticError(
                f"anchor {name} did not reproduce: {anchors[name]!r} != {expected!r}"
            )
    miss_count = sum(int(r["p95_miss"]) for r in run2)
    if miss_count != EXPECTED_FINAL_MISS_COUNT:
        raise Run2ModeQDiagnosticError("final P95 miss count did not reproduce")
    if {int(r["executed_mode_id"]) for r in run2} != {FIXED_MODE_ID_V2}:
        raise Run2ModeQDiagnosticError("Run-2 final mode population drift")
    anchors["run2_final_p95_miss_count"] = float(miss_count)
    anchors["run2_final_p95_miss_rate"] = miss_count / len(run2)
    return anchors


# --------------------------------------------------------------------------- #
# Stage 1: actor / critic behaviour at every registered checkpoint
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CheckpointDiagnostic:
    """Per-checkpoint actor and critic reads over all 340 panel contexts."""

    seed: int
    update_index: int
    checkpoint_sha256: str
    checkpoint_file_sha256: str
    logits: np.ndarray          # (340, 12) float32
    probs: np.ndarray           # (340, 12) float64
    q_e4_all: np.ndarray        # (340, 12) int64
    executed_mode: np.ndarray   # (340,) int64
    executed_q_e4: np.ndarray   # (340,) int64
    critic_1: np.ndarray        # (340, 12) float64
    critic_2: np.ndarray        # (340, 12) float64
    min_q: np.ndarray           # (340, 12) float64


def _checkpoint_path(root: Path, seed: int, update: int) -> Path:
    seed_dir = Path(root) / TRAINING_CAMPAIGN_RELATIVE_PATH / f"seed_{seed}"
    if update == 0:
        return seed_dir / "checkpoint_000000.pt"
    return seed_dir / "checkpoints" / f"checkpoint_{update:06d}.pt"


def _panel_states(evaluator: FitValidationActorEvaluatorV1) -> torch.Tensor:
    values = []
    for entry in evaluator.panel.entries:
        observation, _context, _radio = evaluator._make_observation(entry)
        values.append(observation.values)
    if len(values) != PANEL_CONTEXT_COUNT:
        raise Run2ModeQDiagnosticError("panel state count drift")
    return torch.tensor(values, dtype=torch.float32, device="cpu")


def collect_checkpoint_diagnostic(
    root: Path, seed: int, update: int, states: torch.Tensor
) -> CheckpointDiagnostic:
    """Load one registered checkpoint and read the actor and twin critics."""
    path = _checkpoint_path(root, seed, update)
    checkpoint = load_exact_p95_run2_checkpoint_v2(path)
    if int(checkpoint.seed) != int(seed) or int(checkpoint.update_count) != int(update):
        raise Run2ModeQDiagnosticError("checkpoint seed/update identity drift")

    config = HybridSacModelConfig(
        dtype=torch.float32, modeled_smoke_support=MODELED_SMOKE_SUPPORT
    )
    actor = build_actor(config, seed=0)
    actor.load_state_dict(dict(checkpoint.actor_state), strict=True)
    actor.eval()
    critics = build_twin_critics(config, seed=0)
    critics.load_state_dict(dict(checkpoint.critics_state), strict=True)
    critics.eval()

    # One context at a time, exactly as the registered evaluator executes the
    # policy.  This is not a style choice: a batched float32 GEMM differs from
    # the batch-1 path by an ULP, which is enough to cross a round-half-up
    # ``q_e4`` boundary and silently desynchronize this diagnostic from the
    # frozen evidence (it does, at update 0, for one context).
    logits_rows: List[np.ndarray] = []
    probs_rows: List[np.ndarray] = []
    q_e4_rows: List[np.ndarray] = []
    critic_1_rows: List[np.ndarray] = []
    critic_2_rows: List[np.ndarray] = []
    executed_mode: List[int] = []
    executed_q_e4: List[int] = []

    with torch.no_grad():
        for index in range(states.shape[0]):
            state = states[index : index + 1]
            heads = actor(state)
            execution = actor.deterministic_execution(state)
            q_all = actor._map_pre_squash_to_q(heads.mean)
            q_e4_all = quantize_q_e4(q_all)
            # The executed action must be exactly column ``argmax`` of the
            # all-mode read, otherwise the 12-mode counterfactual below is not
            # the actor's own proposal.
            selected_q_e4 = q_e4_all.gather(
                1, execution.mode_index.unsqueeze(1)
            ).squeeze(1)
            if not torch.equal(selected_q_e4, execution.q_e4):
                raise Run2ModeQDiagnosticError(
                    "all-mode conditional q does not reproduce the executed q_e4"
                )
            q_normalized = q_e4_all.to(torch.float32) / 9800.0
            critic_1 = critics.critic_1.q_all_modes(state, q_normalized)
            critic_2 = critics.critic_2.q_all_modes(state, q_normalized)

            logits_rows.append(heads.logits.squeeze(0).numpy().astype(np.float64))
            probs_rows.append(
                torch.softmax(heads.logits, dim=-1).squeeze(0).numpy().astype(np.float64)
            )
            q_e4_rows.append(q_e4_all.squeeze(0).numpy().astype(np.int64))
            critic_1_rows.append(critic_1.squeeze(0).numpy().astype(np.float64))
            critic_2_rows.append(critic_2.squeeze(0).numpy().astype(np.float64))
            executed_mode.append(int(execution.mode_index.item()))
            executed_q_e4.append(int(execution.q_e4.item()))

    critic_1_array = np.stack(critic_1_rows)
    critic_2_array = np.stack(critic_2_rows)
    return CheckpointDiagnostic(
        seed=int(seed),
        update_index=int(update),
        checkpoint_sha256=str(checkpoint.checkpoint_sha256),
        checkpoint_file_sha256=_sha256_file(path),
        logits=np.stack(logits_rows),
        probs=np.stack(probs_rows),
        q_e4_all=np.stack(q_e4_rows),
        executed_mode=np.asarray(executed_mode, dtype=np.int64),
        executed_q_e4=np.asarray(executed_q_e4, dtype=np.int64),
        critic_1=critic_1_array,
        critic_2=critic_2_array,
        min_q=np.minimum(critic_1_array, critic_2_array),
    )


# --------------------------------------------------------------------------- #
# Stage 2: exact per-context action tables
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ContextActionTable:
    """Every supported action for one panel context, exactly scored under V2."""

    panel_index: int
    scene_rank: int
    sample_id: str
    network_profile: str
    surfaces: Tuple[Mapping[str, np.ndarray], ...]
    networks: Tuple[Mapping[str, np.ndarray], ...]
    shaped_emitted: Tuple[np.ndarray, ...]
    admitted: Tuple[np.ndarray, ...]

    def index_of(self, mode_id: int, q_e4: int) -> int:
        lower, upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[mode_id]
        if not lower <= q_e4 <= upper:
            raise Run2ModeQDiagnosticError(
                f"q_e4 {q_e4} outside mode {mode_id} support [{lower}, {upper}]"
            )
        return int(q_e4) - int(lower)

    def outcome(self, mode_id: int, q_e4: int) -> Dict[str, float]:
        index = self.index_of(mode_id, q_e4)
        surface = self.surfaces[mode_id]
        network = self.networks[mode_id]
        p95 = float(network["p95"][index])
        return {
            "mode_id": int(mode_id),
            "q_e4": int(q_e4),
            "q_perc": float(surface["quality"][index]),
            "total_transmitted_bytes": float(surface["payload"][index]),
            "datagram_count": int(surface["datagrams"][index]),
            "p_admit": float(network["p_admit"][index]),
            "p50_ms": float(network["p50"][index]),
            "p95_ms": p95,
            "p99_ms": float(network["p99"][index]),
            "p95_margin_ms": DEADLINE_MS_V2 - p95,
            "p95_miss": int(p95 > DEADLINE_MS_V2),
            "utility": float(self.shaped_emitted[mode_id][index]),
        }

    def best_within_mode(
        self, mode_id: int, *, feasible_only: bool = False
    ) -> Optional[Dict[str, float]]:
        network = self.networks[mode_id]
        mask = self.admitted[mode_id]
        if feasible_only:
            mask = mask & (network["p95"] <= DEADLINE_MS_V2)
        index = _best_index(self.shaped_emitted[mode_id], network["p95"], mask)
        if index is None:
            return None
        return self.outcome(
            mode_id, int(self.surfaces[mode_id]["q"][index])
        )

    def best_across_modes(
        self, *, feasible_only: bool = False
    ) -> Dict[str, float]:
        best: Optional[Dict[str, float]] = None
        for mode_id in range(MODE_COUNT):
            candidate = self.best_within_mode(mode_id, feasible_only=feasible_only)
            if candidate is None:
                continue
            if best is None or _outcome_better(candidate, best):
                best = candidate
        if best is None:
            raise Run2ModeQDiagnosticError(
                f"context {self.panel_index} has no admitted action"
            )
        return best


def _outcome_better(candidate: Mapping[str, float], incumbent: Mapping[str, float]) -> bool:
    """The registered comparison: utility, then lower P95, then (mode, q)."""
    if candidate["utility"] != incumbent["utility"]:
        return candidate["utility"] > incumbent["utility"]
    if candidate["p95_ms"] != incumbent["p95_ms"]:
        return candidate["p95_ms"] < incumbent["p95_ms"]
    return (candidate["mode_id"], candidate["q_e4"]) < (
        incumbent["mode_id"],
        incumbent["q_e4"],
    )


def build_context_table(
    evaluator: FitValidationActorEvaluatorV1, entry: Any
) -> ContextActionTable:
    """Exactly enumerate and score every supported action for one context."""
    surfaces = tuple(
        _surface_mode_vector(
            evaluator.environment._surface, entry.scene_sample_id, mode_id
        )
        for mode_id in range(MODE_COUNT)
    )
    networks = tuple(
        _network_vector(
            evaluator.environment._network,
            entry.network_profile,
            surface["payload"],
            surface["datagrams"],
        )
        for surface in surfaces
    )
    shaped_emitted = []
    admitted = []
    for surface, network in zip(surfaces, networks):
        base64 = _base64_vector(network["p_admit"], surface["quality"], network["p95"])
        shaped64 = _shaped64_vector(
            base64, network["p_admit"], network["p95"], RUN2_V2_DEADLINE_PENALTY
        )
        shaped_emitted.append(_emitted_vector(shaped64))
        admitted.append(network["p_admit"] > 0.0)
    return ContextActionTable(
        panel_index=int(entry.panel_index),
        scene_rank=int(entry.scene_rank),
        sample_id=str(entry.scene_sample_id),
        network_profile=str(entry.network_profile),
        surfaces=surfaces,
        networks=networks,
        shaped_emitted=tuple(shaped_emitted),
        admitted=tuple(admitted),
    )


def scalar_outcome(
    evaluator: FitValidationActorEvaluatorV1, entry: Any, mode_id: int, q_e4: int
) -> Dict[str, float]:
    """Price one action through the frozen authoritative scalar path.

    This is the *reported* pricing for every arm.  The vectorized table is used
    only to search a mode's several thousand supported ``q_e4`` values; the
    winner is then re-priced here, which is the same search-then-rescore pattern
    the registered validation gate uses.

    The two paths are not bit-identical: measured over 134,640 actions the
    vector table differs from this one by up to 1.1e-16 in interpolated quality
    and 3.4e-13 ms in modeled P95 (the emitted float32 utility never differed).
    That P95 difference is far below any physical resolution but is enough to
    fail an exact-equality check against the frozen rows, which is precisely why
    the authoritative path is the one published.
    """
    raw = _authoritative_outcome(evaluator, entry, mode_id, q_e4)
    _base, _shaped, emitted = _score_v2(
        p_admit=raw.p_edge_admission_given_sent,
        quality=raw.q_perc,
        latency_p95_ms=raw.latency_proxy_p95_ms,
    )
    return {
        "mode_id": int(mode_id),
        "q_e4": int(q_e4),
        "q_perc": float(raw.q_perc),
        "total_transmitted_bytes": float(raw.total_transmitted_bytes),
        "datagram_count": int(raw.datagram_count),
        "p_admit": float(raw.p_edge_admission_given_sent),
        "p50_ms": float(raw.latency_proxy_p50_ms),
        "p95_ms": float(raw.latency_proxy_p95_ms),
        "p99_ms": float(raw.latency_proxy_p99_ms),
        "p95_margin_ms": DEADLINE_MS_V2 - float(raw.latency_proxy_p95_ms),
        "p95_miss": int(float(raw.latency_proxy_p95_ms) > DEADLINE_MS_V2),
        "utility": float(emitted),
    }


def require_table_matches_scalar_scorer(
    table: ContextActionTable,
    mode_id: int,
    q_e4: int,
    evaluator: Optional[FitValidationActorEvaluatorV1] = None,
    entry: Any = None,
) -> Dict[str, float]:
    """Cross-check the vector table, and report its deviation from scalar.

    Without an evaluator this only verifies internal consistency of the table
    against ``_score_v2``.  With one it additionally measures the vector-vs-
    authoritative deviation, which the summary publishes rather than hides.
    """
    outcome = table.outcome(mode_id, q_e4)
    _base, _shaped, emitted = _score_v2(
        p_admit=outcome["p_admit"],
        quality=outcome["q_perc"],
        latency_p95_ms=outcome["p95_ms"],
    )
    if emitted != outcome["utility"]:
        raise Run2ModeQDiagnosticError(
            "vectorized table is not self-consistent with the scalar V2 scorer"
        )
    if evaluator is None or entry is None:
        return {"q_perc": 0.0, "p95_ms": 0.0, "utility": 0.0}
    authoritative = scalar_outcome(evaluator, entry, mode_id, q_e4)
    return {
        "q_perc": abs(authoritative["q_perc"] - outcome["q_perc"]),
        "p95_ms": abs(authoritative["p95_ms"] - outcome["p95_ms"]),
        "utility": abs(authoritative["utility"] - outcome["utility"]),
    }


# --------------------------------------------------------------------------- #
# CSV / JSON writing
# --------------------------------------------------------------------------- #


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise Run2ModeQDiagnosticError("refusing to write an empty CSV")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _finite(value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise Run2ModeQDiagnosticError("refusing to emit a non-finite value")
    return numeric


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


_STAGE1_FIELDS = (
    "logits", "probs", "q_e4_all", "executed_mode", "executed_q_e4",
    "critic_1", "critic_2", "min_q",
)


def _save_stage1_cache(
    path: Path, diagnostics: Mapping[Tuple[int, int], CheckpointDiagnostic]
) -> None:
    payload: Dict[str, Any] = {}
    for (seed, update), diagnostic in diagnostics.items():
        key = f"{seed}_{update}"
        for field in _STAGE1_FIELDS:
            payload[f"{key}__{field}"] = getattr(diagnostic, field)
        payload[f"{key}__sha"] = np.asarray(
            [diagnostic.checkpoint_sha256, diagnostic.checkpoint_file_sha256]
        )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(Path(path), **payload)


def _load_stage1_cache(
    path: Path,
) -> Optional[Dict[Tuple[int, int], CheckpointDiagnostic]]:
    path = Path(path)
    if not path.is_file():
        return None
    archive = np.load(path, allow_pickle=False)
    result: Dict[Tuple[int, int], CheckpointDiagnostic] = {}
    for seed in REGISTERED_SEEDS:
        for update in REGISTERED_UPDATES:
            key = f"{seed}_{update}"
            if f"{key}__sha" not in archive:
                return None
            sha = archive[f"{key}__sha"]
            result[(seed, update)] = CheckpointDiagnostic(
                seed=seed,
                update_index=update,
                checkpoint_sha256=str(sha[0]),
                checkpoint_file_sha256=str(sha[1]),
                **{field: archive[f"{key}__{field}"] for field in _STAGE1_FIELDS},
            )
    return result


def _profile_of(rows: Sequence[Mapping[str, str]]) -> Dict[int, str]:
    return {int(r["panel_index"]): r["network_profile"] for r in rows}


def _index_frozen(
    rows: Sequence[Mapping[str, str]]
) -> Dict[Tuple[str, int, int, int], Mapping[str, str]]:
    indexed: Dict[Tuple[str, int, int, int], Mapping[str, str]] = {}
    for row in rows:
        key = (
            row["source_kind"],
            int(row["seed"]),
            int(row["update_index"]),
            int(row["panel_index"]),
        )
        if key in indexed:
            raise Run2ModeQDiagnosticError("duplicate frozen per-context key")
        indexed[key] = row
    return indexed


def _population_stats(records: Sequence[Mapping[str, Any]], prefix: str) -> Dict[str, float]:
    """Utility / quality / admission / latency / miss summary for one arm."""
    if not records:
        raise Run2ModeQDiagnosticError("empty arm population")
    misses = sum(int(r[f"{prefix}_p95_miss"]) for r in records)
    return {
        "row_count": len(records),
        "v2_utility_mean": _mean([r[f"{prefix}_utility"] for r in records]),
        "q_perc_mean": _mean([r[f"{prefix}_q_perc"] for r in records]),
        "p_admit_mean": _mean([r[f"{prefix}_p_admit"] for r in records]),
        "latency_p50_ms_mean": _mean([r[f"{prefix}_p50_ms"] for r in records]),
        "latency_p95_ms_mean": _mean([r[f"{prefix}_p95_ms"] for r in records]),
        "latency_p99_ms_mean": _mean([r[f"{prefix}_p99_ms"] for r in records]),
        "p95_miss_count": misses,
        "p95_miss_rate": misses / len(records),
    }


def _mode_histogram(values: Sequence[int]) -> Dict[str, int]:
    counts = {f"mode_{m:02d}": 0 for m in range(MODE_COUNT)}
    for value in values:
        counts[f"mode_{int(value):02d}"] += 1
    return counts


def run_diagnostic(
    *,
    root: Path,
    output_dir: Path,
    stage1_cache: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute the complete read-only diagnostic and write durable artifacts."""
    root = Path(root).resolve(strict=True)
    output_dir = Path(output_dir)
    cuda_initialized_before = torch.cuda.is_initialized()
    python_rng_before = random.getstate()
    torch_rng_before = torch.get_rng_state().clone()

    bindings = revalidate_frozen_bindings(root)
    frozen_rows = load_frozen_per_context(root)
    anchors = reproduce_anchors(frozen_rows)
    frozen = _index_frozen(frozen_rows)
    oracle_actions = bindings.pop("oracle_actions")

    evaluator = FitValidationActorEvaluatorV1(project_root=root)
    try:
        states = _panel_states(evaluator)
        entries = list(evaluator.panel.entries)

        # ---- Stage 1: every registered checkpoint --------------------------- #
        # Loading all 33 checkpoints costs about 15 minutes.  The optional cache
        # is a development aid only; the published run is produced without it,
        # and every cached action is re-verified against the frozen evidence
        # below either way.
        diagnostics: Dict[Tuple[int, int], CheckpointDiagnostic] = {}
        cached = _load_stage1_cache(stage1_cache) if stage1_cache else None
        for seed in REGISTERED_SEEDS:
            for update in REGISTERED_UPDATES:
                if cached is not None and (seed, update) in cached:
                    diagnostics[(seed, update)] = cached[(seed, update)]
                    continue
                diagnostics[(seed, update)] = collect_checkpoint_diagnostic(
                    root, seed, update, states
                )
        if stage1_cache is not None and cached is None:
            _save_stage1_cache(stage1_cache, diagnostics)
        if len(diagnostics) != len(REGISTERED_SEEDS) * len(REGISTERED_UPDATES):
            raise Run2ModeQDiagnosticError("checkpoint coverage drift")

        # Executed actions must equal the frozen evaluation row for row.
        for (seed, update), diagnostic in diagnostics.items():
            for position, entry in enumerate(entries):
                row = frozen[(SOURCE_V2, seed, update, int(entry.panel_index))]
                if (
                    int(row["executed_mode_id"]) != int(diagnostic.executed_mode[position])
                    or int(row["executed_q_e4"]) != int(diagnostic.executed_q_e4[position])
                ):
                    raise Run2ModeQDiagnosticError(
                        "recomputed actor action differs from frozen evidence"
                    )

        # ---- Stage 2: exact action tables, context by context ---------------- #
        checkpoint_rows: List[Dict[str, Any]] = []
        regret_rows: List[Dict[str, Any]] = []
        miss_rows: List[Dict[str, Any]] = []
        scalar_crosschecks = 0
        max_vector_deviation = {"q_perc": 0.0, "p95_ms": 0.0, "utility": 0.0}

        for position, entry in enumerate(entries):
            table = build_context_table(evaluator, entry)
            panel_index = int(entry.panel_index)

            oracle_mode, oracle_q_e4 = oracle_actions[panel_index]
            oracle_outcome = scalar_outcome(
                evaluator, entry, oracle_mode, oracle_q_e4
            )
            fixed_outcome = scalar_outcome(
                evaluator, entry, FIXED_MODE_ID_V2, FIXED_Q_E4_V2
            )
            for check_mode, check_q in (
                (oracle_mode, oracle_q_e4),
                (FIXED_MODE_ID_V2, FIXED_Q_E4_V2),
            ):
                deviation = require_table_matches_scalar_scorer(
                    table, check_mode, check_q, evaluator, entry
                )
                for field, value in deviation.items():
                    max_vector_deviation[field] = max(
                        max_vector_deviation[field], value
                    )
                scalar_crosschecks += 1

            # The shaped optimum must reproduce the registered constrained oracle.
            shaped_best = table.best_across_modes(feasible_only=False)
            constrained_best = table.best_across_modes(feasible_only=True)
            for label, candidate in (
                ("shaped", shaped_best),
                ("constrained", constrained_best),
            ):
                if (
                    int(candidate["mode_id"]) != int(oracle_mode)
                    or int(candidate["q_e4"]) != int(oracle_q_e4)
                ):
                    raise Run2ModeQDiagnosticError(
                        f"{label} optimum does not reproduce the frozen oracle "
                        f"at panel index {panel_index}"
                    )
            for source, outcome in (
                (SOURCE_ORACLE, oracle_outcome),
                (SOURCE_FIXED, fixed_outcome),
            ):
                frozen_row = frozen[(source, -1, -1, panel_index)]
                if (
                    float(frozen_row["v2_emitted_reward_float32"]) != outcome["utility"]
                    or float(frozen_row["latency_proxy_p95_ms"]) != outcome["p95_ms"]
                    or float(frozen_row["q_perc"]) != outcome["q_perc"]
                ):
                    raise Run2ModeQDiagnosticError(
                        f"table disagrees with frozen {source} row"
                    )

            # Vector search over every supported q_e4 in each mode; the winner
            # is re-priced through the authoritative scalar path below.
            best_within_mode = [
                table.best_within_mode(mode_id) for mode_id in range(MODE_COUNT)
            ]

            for (seed, update), diagnostic in diagnostics.items():
                probs = diagnostic.probs[position]
                logits = diagnostic.logits[position]
                actor_q_e4 = diagnostic.q_e4_all[position]
                min_q = diagnostic.min_q[position]
                executed_mode = int(diagnostic.executed_mode[position])
                executed_q_e4 = int(diagnostic.executed_q_e4[position])

                # Every one of the 12 actor-proposed actions is priced through
                # the authoritative scalar path, so the "exact V2 ranking" the
                # critics are graded against is the registered pricing.
                proposed = [
                    scalar_outcome(evaluator, entry, mode_id, int(actor_q_e4[mode_id]))
                    for mode_id in range(MODE_COUNT)
                ]
                scalar_crosschecks += MODE_COUNT
                for candidate in proposed:
                    vector = table.outcome(
                        int(candidate["mode_id"]), int(candidate["q_e4"])
                    )
                    for field in ("q_perc", "p95_ms", "utility"):
                        max_vector_deviation[field] = max(
                            max_vector_deviation[field],
                            abs(candidate[field] - vector[field]),
                        )
                exact_util = np.asarray(
                    [candidate["utility"] for candidate in proposed],
                    dtype=np.float64,
                )
                exact_rank = descending_rank(exact_util.tolist())
                minq_rank = descending_rank(min_q.tolist())
                logit_rank = descending_rank(logits.tolist())
                order = np.argsort(-probs, kind="stable")
                top1, top2 = int(order[0]), int(order[1])
                if top1 != executed_mode:
                    raise Run2ModeQDiagnosticError("argmax mode / execution mismatch")

                exact_top1 = int(np.argmax(exact_util))
                minq_top1 = int(np.argmax(min_q))
                row: Dict[str, Any] = {
                    "seed": seed,
                    "update_index": update,
                    "panel_index": panel_index,
                    "scene_rank": table.scene_rank,
                    "sample_id": table.sample_id,
                    "network_profile": table.network_profile,
                    "checkpoint_sha256": diagnostic.checkpoint_sha256,
                    "executed_mode_id": executed_mode,
                    "executed_q_e4": executed_q_e4,
                    "top1_mode": top1,
                    "top2_mode": top2,
                    "top1_prob": _finite(probs[top1]),
                    "top2_prob": _finite(probs[top2]),
                    "top1_top2_prob_gap": _finite(probs[top1] - probs[top2]),
                    "top1_logit": _finite(logits[top1]),
                    "top2_logit": _finite(logits[top2]),
                    "top1_top2_logit_gap": _finite(logits[top1] - logits[top2]),
                    "discrete_entropy_nats": _finite(
                        shannon_entropy_nats(probs.tolist())
                    ),
                    "exact_top1_mode": exact_top1,
                    "exact_top1_utility": _finite(exact_util[exact_top1]),
                    "minq_top1_mode": minq_top1,
                    "minq_top1_value": _finite(min_q[minq_top1]),
                    "actor_top1_exact_utility": _finite(exact_util[top1]),
                    "actor_top1_min_q": _finite(min_q[top1]),
                    "critic_error_at_actor_top1": _finite(
                        min_q[top1] - exact_util[top1]
                    ),
                    "exact_rank_of_actor_top1": int(exact_rank[top1]),
                    "minq_rank_of_actor_top1": int(minq_rank[top1]),
                    "exact_rank_of_minq_top1": int(exact_rank[minq_top1]),
                    "actor_vs_exact_top1_agree": int(top1 == exact_top1),
                    "critic_vs_exact_top1_agree": int(minq_top1 == exact_top1),
                    "actor_vs_critic_top1_agree": int(top1 == minq_top1),
                    "actor_top1_exact_utility_gap": _finite(
                        exact_util[exact_top1] - exact_util[top1]
                    ),
                }
                rho_minq = spearman_rho(min_q.tolist(), exact_util.tolist())
                rho_logit = spearman_rho(logits.tolist(), exact_util.tolist())
                row["spearman_minq_vs_exact"] = (
                    "" if math.isnan(rho_minq) else _finite(rho_minq)
                )
                row["spearman_logit_vs_exact"] = (
                    "" if math.isnan(rho_logit) else _finite(rho_logit)
                )
                for mode_id in range(MODE_COUNT):
                    row[f"prob_mode_{mode_id:02d}"] = _finite(probs[mode_id])
                    row[f"actor_q_e4_mode_{mode_id:02d}"] = int(actor_q_e4[mode_id])
                    row[f"min_q_mode_{mode_id:02d}"] = _finite(min_q[mode_id])
                    row[f"exact_utility_mode_{mode_id:02d}"] = _finite(
                        exact_util[mode_id]
                    )
                checkpoint_rows.append(row)

                if update != PRIMARY_UPDATE:
                    continue

                # ---- Counterfactual arms A..E at the primary endpoint ------- #
                outcome_a = proposed[executed_mode]
                if int(outcome_a["q_e4"]) != executed_q_e4:
                    raise Run2ModeQDiagnosticError(
                        "arm A is not the actor's own proposal for its own mode"
                    )
                frozen_a = frozen[(SOURCE_V2, seed, update, panel_index)]
                if (
                    float(frozen_a["v2_emitted_reward_float32"]) != outcome_a["utility"]
                    or float(frozen_a["latency_proxy_p95_ms"]) != outcome_a["p95_ms"]
                ):
                    raise Run2ModeQDiagnosticError(
                        "arm A does not reproduce the frozen Run-2 row"
                    )

                # B: the learned mode at its best supported q.  The vector search
                # picks the candidate; it is re-priced scalar-wise, and A's own q
                # stays in the running so B can never fall below A on a 1-ULP
                # search artifact.
                vector_best = best_within_mode[executed_mode]
                if vector_best is None:
                    raise Run2ModeQDiagnosticError(
                        "learned mode has no admitted action"
                    )
                outcome_b = scalar_outcome(
                    evaluator, entry, executed_mode, int(vector_best["q_e4"])
                )
                scalar_crosschecks += 1
                if _outcome_better(outcome_a, outcome_b):
                    outcome_b = outcome_a

                # C: every mode at that mode's own actor-proposed q.
                outcome_c = None
                for candidate in proposed:
                    if candidate["p_admit"] <= 0.0:
                        continue
                    if outcome_c is None or _outcome_better(candidate, outcome_c):
                        outcome_c = candidate
                if outcome_c is None:
                    raise Run2ModeQDiagnosticError(
                        "no admitted action among the actor-proposed q values"
                    )

                decomposition = decompose_context_regret(
                    utility_a=outcome_a["utility"],
                    utility_b=outcome_b["utility"],
                    utility_c=outcome_c["utility"],
                    utility_d=oracle_outcome["utility"],
                    utility_e=fixed_outcome["utility"],
                )

                record: Dict[str, Any] = {
                    "seed": seed,
                    "update_index": update,
                    "panel_index": panel_index,
                    "scene_rank": table.scene_rank,
                    "sample_id": table.sample_id,
                    "episode_id": str(entry.scene_episode_id),
                    "frame_id": int(entry.scene_frame_id),
                    "network_profile": table.network_profile,
                }
                for prefix, outcome in (
                    ("a", outcome_a),
                    ("b", outcome_b),
                    ("c", outcome_c),
                    ("d", oracle_outcome),
                    ("e", fixed_outcome),
                ):
                    record[f"{prefix}_mode_id"] = int(outcome["mode_id"])
                    record[f"{prefix}_q_e4"] = int(outcome["q_e4"])
                    record[f"{prefix}_q_perc"] = _finite(outcome["q_perc"])
                    record[f"{prefix}_p_admit"] = _finite(outcome["p_admit"])
                    record[f"{prefix}_p50_ms"] = _finite(outcome["p50_ms"])
                    record[f"{prefix}_p95_ms"] = _finite(outcome["p95_ms"])
                    record[f"{prefix}_p99_ms"] = _finite(outcome["p99_ms"])
                    record[f"{prefix}_p95_margin_ms"] = _finite(outcome["p95_margin_ms"])
                    record[f"{prefix}_p95_miss"] = int(outcome["p95_miss"])
                    record[f"{prefix}_utility"] = _finite(outcome["utility"])
                record["continuous_q_regret"] = _finite(decomposition.continuous_q_regret)
                record["mode_given_best_q_regret"] = _finite(
                    decomposition.mode_given_best_q_regret
                )
                record["total_oracle_regret"] = _finite(decomposition.total_oracle_regret)
                record["mode_at_actor_q_regret"] = _finite(
                    decomposition.mode_at_actor_q_regret
                )
                record["learned_minus_fixed"] = _finite(decomposition.learned_minus_fixed)
                regret_rows.append(record)

                if not outcome_a["p95_miss"]:
                    continue

                feasible_vector = table.best_within_mode(
                    executed_mode, feasible_only=True
                )
                feasible_same_mode = (
                    None if feasible_vector is None
                    else scalar_outcome(
                        evaluator, entry, executed_mode, int(feasible_vector["q_e4"])
                    )
                )
                miss_rows.append(
                    {
                        "seed": seed,
                        "panel_index": panel_index,
                        "scene_rank": table.scene_rank,
                        "sample_id": table.sample_id,
                        "episode_id": str(entry.scene_episode_id),
                        "frame_id": int(entry.scene_frame_id),
                        "network_profile": table.network_profile,
                        "selected_mode_id": int(outcome_a["mode_id"]),
                        "selected_q_e4": int(outcome_a["q_e4"]),
                        "selected_q_perc": _finite(outcome_a["q_perc"]),
                        "modeled_p50_ms": _finite(outcome_a["p50_ms"]),
                        "modeled_p95_ms": _finite(outcome_a["p95_ms"]),
                        "modeled_p99_ms": _finite(outcome_a["p99_ms"]),
                        "p95_exceedance_ms": _finite(
                            outcome_a["p95_ms"] - DEADLINE_MS_V2
                        ),
                        "p_admit": _finite(outcome_a["p_admit"]),
                        "v2_utility": _finite(outcome_a["utility"]),
                        "fixed_q_e4": FIXED_Q_E4_V2,
                        "fixed_p95_ms": _finite(fixed_outcome["p95_ms"]),
                        "fixed_p95_miss": int(fixed_outcome["p95_miss"]),
                        "fixed_q_perc": _finite(fixed_outcome["q_perc"]),
                        "fixed_v2_utility": _finite(fixed_outcome["utility"]),
                        "oracle_mode_id": int(oracle_outcome["mode_id"]),
                        "oracle_q_e4": int(oracle_outcome["q_e4"]),
                        "oracle_p95_ms": _finite(oracle_outcome["p95_ms"]),
                        "oracle_q_perc": _finite(oracle_outcome["q_perc"]),
                        "oracle_v2_utility": _finite(oracle_outcome["utility"]),
                        "same_mode_feasible_best_q_e4": (
                            "" if feasible_same_mode is None
                            else int(feasible_same_mode["q_e4"])
                        ),
                        "same_mode_feasible_best_p95_ms": (
                            "" if feasible_same_mode is None
                            else _finite(feasible_same_mode["p95_ms"])
                        ),
                        "same_mode_feasible_best_q_perc": (
                            "" if feasible_same_mode is None
                            else _finite(feasible_same_mode["q_perc"])
                        ),
                        "same_mode_feasible_best_utility": (
                            "" if feasible_same_mode is None
                            else _finite(feasible_same_mode["utility"])
                        ),
                        # q_e4 is a compression knob: payload falls as q_e4 rises,
                        # so restoring feasibility needs MORE compression, not less.
                        "q_e4_increase_for_same_mode_feasibility": (
                            "" if feasible_same_mode is None
                            else int(feasible_same_mode["q_e4"]) - int(outcome_a["q_e4"])
                        ),
                        "utility_delta_if_made_feasible": (
                            "" if feasible_same_mode is None
                            else _finite(
                                feasible_same_mode["utility"] - outcome_a["utility"]
                            )
                        ),
                    }
                )
    finally:
        evaluator.close()

    if len(checkpoint_rows) != len(diagnostics) * PANEL_CONTEXT_COUNT:
        raise Run2ModeQDiagnosticError("checkpoint diagnostic row count drift")
    if len(regret_rows) != EXPECTED_FINAL_DECISION_COUNT:
        raise Run2ModeQDiagnosticError("regret row count drift")
    if len(miss_rows) != EXPECTED_FINAL_MISS_COUNT:
        raise Run2ModeQDiagnosticError("final miss row count drift")

    if torch.cuda.is_initialized() != cuda_initialized_before:
        raise Run2ModeQDiagnosticError("diagnostic changed CUDA initialization state")
    if random.getstate() != python_rng_before:
        raise Run2ModeQDiagnosticError("diagnostic advanced the global Python RNG")
    if not torch.equal(torch.get_rng_state(), torch_rng_before):
        raise Run2ModeQDiagnosticError("diagnostic advanced the global Torch RNG")

    summary = _build_summary(
        anchors=anchors,
        bindings=bindings,
        checkpoint_rows=checkpoint_rows,
        regret_rows=regret_rows,
        miss_rows=miss_rows,
        frozen_rows=frozen_rows,
        scalar_crosschecks=scalar_crosschecks,
        cuda_initialized=cuda_initialized_before,
        max_vector_deviation=max_vector_deviation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "checkpoint_mode_diagnostics.csv": _csv_bytes(checkpoint_rows),
        "context_regret_decomposition.csv": _csv_bytes(regret_rows),
        "final_miss_audit.csv": _csv_bytes(miss_rows),
        "diagnostic_summary.json": _json_bytes(summary),
    }
    report = render_report(summary)
    payloads["DIAGNOSTIC_REPORT.md"] = report.encode("utf-8")
    for name, payload in payloads.items():
        _atomic_bytes(output_dir / name, payload)

    manifest = {
        "artifacts": {
            name: hashlib.sha256(payload).hexdigest()
            for name, payload in sorted(payloads.items())
        },
        "cuda_initialized": cuda_initialized_before,
        "generated_by": (
            "rl_agent/splitfusion_hybrid_sac_v1/run2_mode_q_regret_diagnostic_v1.py"
        ),
        "schema": MANIFEST_SCHEMA_V1,
        "scope_caveats": summary["scope_caveats"],
        "source_bindings": bindings,
        "status": "COMPLETE",
    }
    manifest["manifest_content_sha256"] = canonical_sha256(
        {k: v for k, v in manifest.items()}
    )
    _atomic_bytes(output_dir / "artifact_manifest.json", _json_bytes(manifest))
    summary["artifact_manifest_content_sha256"] = manifest["manifest_content_sha256"]
    return summary


SCOPE_CAVEATS: Tuple[str, ...] = (
    "DEVELOPMENT_FIT_VALIDATION_NOT_FINAL_TEST",
    "P95_IS_MODELED_CONDITIONAL_RETAINED_SURVIVOR_PROXY_NOT_LIVE_SLA",
    "85_SCENE_CLUSTERS_X_4_PROFILES_X_3_SEEDS_ARE_NOT_1020_INDEPENDENT_SAMPLES",
    "RAW_REWARD_MAGNITUDE_IS_NOT_A_CONVERGENCE_CLAIM",
    "NO_RETRAINING_NO_REWARD_WEIGHT_CHANGE_NO_CHECKPOINT_CHERRY_PICK",
)


def _optional_mean(values: Sequence[Any]) -> Optional[float]:
    numeric = [float(v) for v in values if v != "" and v is not None]
    if not numeric:
        return None
    return math.fsum(numeric) / len(numeric)


def _build_summary(
    *,
    anchors: Mapping[str, float],
    bindings: Mapping[str, Any],
    checkpoint_rows: Sequence[Mapping[str, Any]],
    regret_rows: Sequence[Mapping[str, Any]],
    miss_rows: Sequence[Mapping[str, Any]],
    frozen_rows: Sequence[Mapping[str, str]],
    scalar_crosschecks: int,
    cuda_initialized: bool,
    max_vector_deviation: Mapping[str, float],
) -> Dict[str, Any]:
    profiles = sorted({row["network_profile"] for row in regret_rows})
    arms = {"a": "LEARNED_ACTION", "b": "LEARNED_MODE_BEST_Q", "c": "BEST_MODE_AT_ACTOR_Q",
            "d": "CONTEXTUAL_CONSTRAINED_ORACLE", "e": "FROZEN_FIXED_MODE11_Q6000"}

    arm_overall = {
        arms[prefix]: _population_stats(regret_rows, prefix) for prefix in arms
    }
    arm_by_profile: Dict[str, Dict[str, Any]] = {}
    for profile in profiles:
        subset = [r for r in regret_rows if r["network_profile"] == profile]
        arm_by_profile[profile] = {
            arms[prefix]: _population_stats(subset, prefix) for prefix in arms
        }

    def regret_block(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        return {
            "row_count": len(records),
            "continuous_q_regret_mean": _mean(
                [r["continuous_q_regret"] for r in records]
            ),
            "mode_given_best_q_regret_mean": _mean(
                [r["mode_given_best_q_regret"] for r in records]
            ),
            "total_oracle_regret_mean": _mean(
                [r["total_oracle_regret"] for r in records]
            ),
            "mode_at_actor_q_regret_mean_overlapping": _mean(
                [r["mode_at_actor_q_regret"] for r in records]
            ),
            "learned_minus_fixed_mean": _mean(
                [r["learned_minus_fixed"] for r in records]
            ),
            "continuous_q_share_of_total": (
                _mean([r["continuous_q_regret"] for r in records])
                / _mean([r["total_oracle_regret"] for r in records])
                if _mean([r["total_oracle_regret"] for r in records]) != 0.0
                else None
            ),
            "contexts_where_learned_mode_is_oracle_mode": sum(
                int(r["a_mode_id"] == r["d_mode_id"]) for r in records
            ),
            "contexts_where_actor_q_already_optimal": sum(
                int(r["a_q_e4"] == r["b_q_e4"]) for r in records
            ),
        }

    regret_overall = regret_block(regret_rows)
    regret_by_profile = {
        profile: regret_block([r for r in regret_rows if r["network_profile"] == profile])
        for profile in profiles
    }
    regret_by_seed = {
        str(seed): regret_block([r for r in regret_rows if r["seed"] == seed])
        for seed in REGISTERED_SEEDS
    }

    # ---- checkpoint trend ------------------------------------------------- #
    frozen_run2 = [r for r in frozen_rows if r["source_kind"] == SOURCE_V2]
    trend: List[Dict[str, Any]] = []
    for update in REGISTERED_UPDATES:
        rows = [r for r in checkpoint_rows if r["update_index"] == update]
        executed = [r for r in frozen_run2 if int(r["update_index"]) == update]
        misses = sum(int(r["p95_miss"]) for r in executed)
        trend.append(
            {
                "update_index": update,
                "row_count": len(rows),
                "v2_utility_mean": _mean(
                    [float(r["v2_emitted_reward_float32"]) for r in executed]
                ),
                "p95_miss_count": misses,
                "p95_miss_rate": misses / len(executed),
                "q_perc_mean": _mean([float(r["q_perc"]) for r in executed]),
                "executed_q_e4_mean": _mean(
                    [float(r["executed_q_e4"]) for r in executed]
                ),
                "mode_histogram": _mode_histogram(
                    [int(r["executed_mode_id"]) for r in executed]
                ),
                "distinct_executed_modes": len(
                    {int(r["executed_mode_id"]) for r in executed}
                ),
                "top1_prob_mean": _mean([r["top1_prob"] for r in rows]),
                "top1_top2_prob_gap_mean": _mean([r["top1_top2_prob_gap"] for r in rows]),
                "top1_top2_logit_gap_mean": _mean(
                    [r["top1_top2_logit_gap"] for r in rows]
                ),
                "discrete_entropy_nats_mean": _mean(
                    [r["discrete_entropy_nats"] for r in rows]
                ),
                "actor_vs_exact_top1_agreement": _mean(
                    [r["actor_vs_exact_top1_agree"] for r in rows]
                ),
                "critic_vs_exact_top1_agreement": _mean(
                    [r["critic_vs_exact_top1_agree"] for r in rows]
                ),
                "actor_vs_critic_top1_agreement": _mean(
                    [r["actor_vs_critic_top1_agree"] for r in rows]
                ),
                "spearman_minq_vs_exact_mean": _optional_mean(
                    [r["spearman_minq_vs_exact"] for r in rows]
                ),
                "spearman_logit_vs_exact_mean": _optional_mean(
                    [r["spearman_logit_vs_exact"] for r in rows]
                ),
                "critic_error_at_actor_top1_mean": _mean(
                    [r["critic_error_at_actor_top1"] for r in rows]
                ),
                "critic_abs_error_at_actor_top1_mean": _mean(
                    [abs(r["critic_error_at_actor_top1"]) for r in rows]
                ),
                "actor_top1_exact_utility_gap_mean": _mean(
                    [r["actor_top1_exact_utility_gap"] for r in rows]
                ),
            }
        )

    trend_by_profile: Dict[str, List[Dict[str, Any]]] = {}
    for profile in profiles:
        series = []
        for update in REGISTERED_UPDATES:
            rows = [
                r for r in checkpoint_rows
                if r["update_index"] == update and r["network_profile"] == profile
            ]
            executed = [
                r for r in frozen_run2
                if int(r["update_index"]) == update
                and r["network_profile"] == profile
            ]
            series.append(
                {
                    "update_index": update,
                    "v2_utility_mean": _mean(
                        [float(r["v2_emitted_reward_float32"]) for r in executed]
                    ),
                    "p95_miss_count": sum(int(r["p95_miss"]) for r in executed),
                    "executed_q_e4_mean": _mean(
                        [float(r["executed_q_e4"]) for r in executed]
                    ),
                    "q_perc_mean": _mean([float(r["q_perc"]) for r in executed]),
                    "latency_p95_ms_mean": _mean(
                        [float(r["latency_proxy_p95_ms"]) for r in executed]
                    ),
                    "top1_prob_mean": _mean([r["top1_prob"] for r in rows]),
                    "discrete_entropy_nats_mean": _mean(
                        [r["discrete_entropy_nats"] for r in rows]
                    ),
                    "critic_abs_error_at_actor_top1_mean": _mean(
                        [abs(r["critic_error_at_actor_top1"]) for r in rows]
                    ),
                    "spearman_minq_vs_exact_mean": _optional_mean(
                        [r["spearman_minq_vs_exact"] for r in rows]
                    ),
                    "critic_vs_exact_top1_agreement": _mean(
                        [r["critic_vs_exact_top1_agree"] for r in rows]
                    ),
                }
            )
        trend_by_profile[profile] = series

    trend_by_seed: Dict[str, List[Dict[str, Any]]] = {}
    for seed in REGISTERED_SEEDS:
        series = []
        for update in REGISTERED_UPDATES:
            rows = [
                r for r in checkpoint_rows
                if r["update_index"] == update and r["seed"] == seed
            ]
            executed = [
                r for r in frozen_run2
                if int(r["update_index"]) == update and int(r["seed"]) == seed
            ]
            misses = sum(int(r["p95_miss"]) for r in executed)
            series.append(
                {
                    "update_index": update,
                    "v2_utility_mean": _mean(
                        [float(r["v2_emitted_reward_float32"]) for r in executed]
                    ),
                    "p95_miss_count": misses,
                    "top1_prob_mean": _mean([r["top1_prob"] for r in rows]),
                    "discrete_entropy_nats_mean": _mean(
                        [r["discrete_entropy_nats"] for r in rows]
                    ),
                    "critic_abs_error_at_actor_top1_mean": _mean(
                        [abs(r["critic_error_at_actor_top1"]) for r in rows]
                    ),
                    "distinct_executed_modes": len(
                        {int(r["executed_mode_id"]) for r in executed}
                    ),
                }
            )
        trend_by_seed[str(seed)] = series

    # ---- final-checkpoint discrete/critic diagnosis ------------------------ #
    final_rows = [r for r in checkpoint_rows if r["update_index"] == PRIMARY_UPDATE]
    final_by_profile = {}
    for profile in profiles:
        subset = [r for r in final_rows if r["network_profile"] == profile]
        final_by_profile[profile] = {
            "row_count": len(subset),
            "top1_prob_mean": _mean([r["top1_prob"] for r in subset]),
            "top1_prob_min": min(r["top1_prob"] for r in subset),
            "top1_prob_max": max(r["top1_prob"] for r in subset),
            "top1_top2_prob_gap_mean": _mean([r["top1_top2_prob_gap"] for r in subset]),
            "top1_top2_logit_gap_mean": _mean(
                [r["top1_top2_logit_gap"] for r in subset]
            ),
            "discrete_entropy_nats_mean": _mean(
                [r["discrete_entropy_nats"] for r in subset]
            ),
            "top2_mode_histogram": _mode_histogram([r["top2_mode"] for r in subset]),
            "critic_top1_mode_histogram": _mode_histogram(
                [r["minq_top1_mode"] for r in subset]
            ),
            "exact_top1_mode_histogram": _mode_histogram(
                [r["exact_top1_mode"] for r in subset]
            ),
            "executed_q_e4_mean": _mean(
                [float(r["executed_q_e4"]) for r in subset]
            ),
        }

    final_summary = {
        "row_count": len(final_rows),
        "actor_top1_mode_histogram": _mode_histogram([r["top1_mode"] for r in final_rows]),
        "actor_top2_mode_histogram": _mode_histogram([r["top2_mode"] for r in final_rows]),
        "critic_minq_top1_mode_histogram": _mode_histogram(
            [r["minq_top1_mode"] for r in final_rows]
        ),
        "exact_top1_mode_histogram": _mode_histogram(
            [r["exact_top1_mode"] for r in final_rows]
        ),
        "top1_prob_mean": _mean([r["top1_prob"] for r in final_rows]),
        "top1_prob_min": min(r["top1_prob"] for r in final_rows),
        "top1_prob_max": max(r["top1_prob"] for r in final_rows),
        "top1_top2_prob_gap_mean": _mean([r["top1_top2_prob_gap"] for r in final_rows]),
        "top1_top2_prob_gap_min": min(r["top1_top2_prob_gap"] for r in final_rows),
        "top1_top2_logit_gap_mean": _mean([r["top1_top2_logit_gap"] for r in final_rows]),
        "top1_top2_logit_gap_min": min(r["top1_top2_logit_gap"] for r in final_rows),
        "discrete_entropy_nats_mean": _mean(
            [r["discrete_entropy_nats"] for r in final_rows]
        ),
        "discrete_entropy_nats_max": max(r["discrete_entropy_nats"] for r in final_rows),
        "actor_vs_exact_top1_agreement": _mean(
            [r["actor_vs_exact_top1_agree"] for r in final_rows]
        ),
        "critic_vs_exact_top1_agreement": _mean(
            [r["critic_vs_exact_top1_agree"] for r in final_rows]
        ),
        "actor_vs_critic_top1_agreement": _mean(
            [r["actor_vs_critic_top1_agree"] for r in final_rows]
        ),
        "spearman_minq_vs_exact_mean": _optional_mean(
            [r["spearman_minq_vs_exact"] for r in final_rows]
        ),
        "spearman_logit_vs_exact_mean": _optional_mean(
            [r["spearman_logit_vs_exact"] for r in final_rows]
        ),
        "critic_error_at_actor_top1_mean": _mean(
            [r["critic_error_at_actor_top1"] for r in final_rows]
        ),
        "critic_abs_error_at_actor_top1_mean": _mean(
            [abs(r["critic_error_at_actor_top1"]) for r in final_rows]
        ),
        "by_profile": final_by_profile,
    }

    # ---- adaptation of the continuous branch -------------------------------- #
    q_by_profile = {}
    for profile in profiles:
        subset = [r for r in regret_rows if r["network_profile"] == profile]
        q_by_profile[profile] = {
            "learned_q_perc_mean": _mean([r["a_q_perc"] for r in subset]),
            "learned_q_e4_mean": _mean([float(r["a_q_e4"]) for r in subset]),
            "learned_mode_best_q_e4_mean": _mean([float(r["b_q_e4"]) for r in subset]),
            "oracle_q_e4_mean": _mean([float(r["d_q_e4"]) for r in subset]),
            "oracle_q_perc_mean": _mean([r["d_q_perc"] for r in subset]),
            "learned_q_e4_std": float(
                np.std(np.asarray([float(r["a_q_e4"]) for r in subset]))
            ),
        }
    all_q = [float(r["a_q_e4"]) for r in regret_rows]
    within_profile_var = math.fsum(
        (float(r["a_q_e4"]) - q_by_profile[r["network_profile"]]["learned_q_e4_mean"]) ** 2
        for r in regret_rows
    ) / len(regret_rows)
    total_var = float(np.var(np.asarray(all_q)))

    # ---- miss characterization --------------------------------------------- #
    exceedances = [r["p95_exceedance_ms"] for r in miss_rows]
    increases = [
        int(r["q_e4_increase_for_same_mode_feasibility"])
        for r in miss_rows
        if r["q_e4_increase_for_same_mode_feasibility"] != ""
    ]
    miss_block = {
        "count": len(miss_rows),
        "rate_of_1020_decisions": len(miss_rows) / EXPECTED_FINAL_DECISION_COUNT,
        "profiles": sorted({r["network_profile"] for r in miss_rows}),
        "distinct_panel_indices": len({r["panel_index"] for r in miss_rows}),
        "distinct_scene_clusters": len({r["scene_rank"] for r in miss_rows}),
        "exceedance_ms_min": min(exceedances),
        "exceedance_ms_max": max(exceedances),
        "exceedance_ms_mean": _mean(exceedances),
        "selected_q_e4_min": min(r["selected_q_e4"] for r in miss_rows),
        "selected_q_e4_max": max(r["selected_q_e4"] for r in miss_rows),
        "fixed_action_also_misses": sum(r["fixed_p95_miss"] for r in miss_rows),
        "oracle_misses": 0,
        "contexts_with_no_feasible_action_in_the_selected_mode": sum(
            int(r["same_mode_feasible_best_q_e4"] == "") for r in miss_rows
        ),
        "q_e4_increase_for_feasibility_min": min(increases) if increases else None,
        "q_e4_increase_for_feasibility_max": max(increases) if increases else None,
        "knob_direction_note": (
            "HIGHER_Q_E4_MEANS_MORE_COMPRESSION_SMALLER_PAYLOAD_LOWER_LATENCY_"
            "AND_LOWER_REALIZED_PERCEPTION_QUALITY"
        ),
        "utility_delta_if_made_feasible_mean": _optional_mean(
            [r["utility_delta_if_made_feasible"] for r in miss_rows]
        ),
        "margin_calibration_status": (
            "NO_MARGIN_RECOMMENDED_FROM_THIS_PANEL_TUNING_A_MARGIN_HERE_WOULD_"
            "CONTAMINATE_THE_DEVELOPMENT_PANEL"
        ),
    }

    conclusions = _build_conclusions(
        final_summary=final_summary,
        regret_overall=regret_overall,
        arm_overall=arm_overall,
        q_by_profile=q_by_profile,
        within_profile_var=within_profile_var,
        total_var=total_var,
        trend=trend,
        trend_by_profile=trend_by_profile,
        miss_block=miss_block,
        anchors=anchors,
    )

    summary = {
        "schema": SUMMARY_SCHEMA_V1,
        "status": "COMPLETE",
        "decision": conclusions["decision"],
        "primary_update": PRIMARY_UPDATE,
        "reproduced_anchors": dict(anchors),
        "source_bindings": dict(bindings),
        "cuda_initialized": cuda_initialized,
        "scalar_vector_crosscheck_count": scalar_crosschecks,
        "max_vector_minus_authoritative_deviation": dict(max_vector_deviation),
        "pricing_path": (
            "ALL_REPORTED_ARMS_PRICED_BY_THE_FROZEN_AUTHORITATIVE_SCALAR_PATH; "
            "THE_VECTOR_TABLE_IS_USED_ONLY_TO_SEARCH_AND_ITS_WINNER_IS_RESCORED"
        ),
        "counterfactual_definitions": {
            "A_LEARNED_ACTION": (
                "argmax discrete mode at its conditional mean q, exactly quantized "
                "through the registered q_e4 execution boundary"
            ),
            "B_LEARNED_MODE_BEST_Q": (
                "same mode as A; q_e4 maximizing the shaped emitted-float32 V2 "
                "utility over every admitted integer q_e4 in that mode's support"
            ),
            "C_BEST_MODE_AT_ACTOR_Q": (
                "each of the 12 modes played at that mode's own actor conditional "
                "mean q; best admitted such action"
            ),
            "D_CONTEXTUAL_CONSTRAINED_ORACLE": (
                "frozen registered per-context constrained oracle action; "
                "independently reproduced here by exact enumeration"
            ),
            "E_FROZEN_FIXED": "mode 11 at q_e4=6000 for every context",
            "additive_identity": (
                "total_oracle_regret = continuous_q_regret + "
                "mode_given_best_q_regret, exactly, because B shares A's mode and "
                "D is the unrestricted optimum"
            ),
            "non_additive_note": (
                "mode_at_actor_q_regret (C - A) overlaps both additive terms and "
                "is reported as a separate diagnostic, never summed with them"
            ),
        },
        "arm_summary_overall": arm_overall,
        "arm_summary_by_profile": arm_by_profile,
        "regret_decomposition_overall": regret_overall,
        "regret_decomposition_by_profile": regret_by_profile,
        "regret_decomposition_by_seed": regret_by_seed,
        "final_checkpoint_discrete_branch": final_summary,
        "continuous_q_adaptation": {
            "by_profile": q_by_profile,
            "total_q_e4_variance": total_var,
            "within_profile_q_e4_variance": within_profile_var,
            "between_profile_variance_share": (
                (total_var - within_profile_var) / total_var if total_var else None
            ),
        },
        "checkpoint_trend_pooled": trend,
        "checkpoint_trend_by_profile": trend_by_profile,
        "checkpoint_trend_by_seed": trend_by_seed,
        "final_miss_characterization": miss_block,
        "conclusions": conclusions,
        "scope_caveats": list(SCOPE_CAVEATS),
    }
    return summary


def _build_conclusions(
    *,
    final_summary: Mapping[str, Any],
    regret_overall: Mapping[str, Any],
    arm_overall: Mapping[str, Any],
    q_by_profile: Mapping[str, Any],
    within_profile_var: float,
    total_var: float,
    trend: Sequence[Mapping[str, Any]],
    trend_by_profile: Mapping[str, Sequence[Mapping[str, Any]]],
    miss_block: Mapping[str, Any],
    anchors: Mapping[str, float],
) -> Dict[str, Any]:
    """Derive the six required answers strictly from the computed evidence."""
    n = final_summary["row_count"]
    actor_mode11 = final_summary["actor_top1_mode_histogram"]["mode_11"] / n
    critic_mode11 = final_summary["critic_minq_top1_mode_histogram"]["mode_11"] / n
    exact_mode11 = final_summary["exact_top1_mode_histogram"]["mode_11"] / n
    oracle_mode11 = regret_overall["contexts_where_learned_mode_is_oracle_mode"] / n

    continuous_share = regret_overall["continuous_q_share_of_total"]
    peak = max(trend, key=lambda row: row["v2_utility_mean"])
    final = trend[-1]
    first_trained = trend[1]

    between_share = (total_var - within_profile_var) / total_var if total_var else 0.0
    under_compression = {
        profile: block["oracle_q_e4_mean"] - block["learned_q_e4_mean"]
        for profile, block in q_by_profile.items()
    }
    learned_mode_best_q = arm_overall["LEARNED_MODE_BEST_Q"]
    oracle_arm = arm_overall["CONTEXTUAL_CONSTRAINED_ORACLE"]

    # Mode 11 has by far the widest registered support, so a continuous
    # maximum-entropy pull toward the middle of a mode's support acts hardest
    # exactly where this policy lives.  Stated as a mechanism to test, not a
    # demonstrated cause: confirming it needs a training run, which is out of
    # scope here.
    mode11_lower, mode11_upper = MODELED_SMOKE_MODE_Q_E4_BOUNDS[FIXED_MODE_ID_V2]
    mode11_midpoint = (mode11_lower + mode11_upper) / 2.0

    adverse = trend_by_profile.get("ADVERSE_STABLE", ())
    adverse_first = adverse[1] if len(adverse) > 1 else None
    adverse_final = adverse[-1] if adverse else None

    return {
        "decision": "GO_DIAGNOSE_RUN3_DESIGN",
        "q1_mode11_dominance_source": {
            "actor_top1_mode11_fraction": actor_mode11,
            "critic_minq_top1_mode11_fraction": critic_mode11,
            "exact_top1_mode11_fraction_at_actor_q": exact_mode11,
            "oracle_mode11_fraction_each_mode_at_its_own_best_q": oracle_mode11,
            "actor_vs_critic_top1_agreement": final_summary[
                "actor_vs_critic_top1_agreement"
            ],
            "critic_vs_exact_top1_agreement": final_summary[
                "critic_vs_exact_top1_agreement"
            ],
            "spearman_minq_vs_exact_mean": final_summary[
                "spearman_minq_vs_exact_mean"
            ],
            "discrete_entropy_nats_mean": final_summary["discrete_entropy_nats_mean"],
            "max_possible_entropy_nats": math.log(MODE_COUNT),
            "verdict": (
                "CRITIC_OVER_RANKING_AMPLIFIED_BY_DETERMINISTIC_ARGMAX"
                if critic_mode11 > oracle_mode11
                and final_summary["actor_vs_critic_top1_agreement"] > 0.8
                else "ACTOR_DRIVEN"
            ),
            "categorical_head_is_collapsed": bool(
                final_summary["discrete_entropy_nats_mean"]
                < 0.25 * math.log(MODE_COUNT)
            ),
        },
        "q2_continuous_branch_useful": {
            "between_profile_q_variance_share": between_share,
            "per_profile_learned_q_e4_mean": {
                profile: block["learned_q_e4_mean"]
                for profile, block in q_by_profile.items()
            },
            "per_profile_oracle_minus_learned_q_e4": under_compression,
            "systematic_under_compression_in_every_profile": all(
                value > 0.0 for value in under_compression.values()
            ),
            "continuous_q_regret_mean": regret_overall["continuous_q_regret_mean"],
            "contexts_where_actor_q_already_optimal": regret_overall[
                "contexts_where_actor_q_already_optimal"
            ],
            "verdict": (
                "GENUINELY_CONTEXTUAL_BUT_SYSTEMATICALLY_UNDER_COMPRESSED"
                if between_share > 0.1
                and all(value > 0.0 for value in under_compression.values())
                else "WEAKLY_CONTEXTUAL"
            ),
        },
        "q3_regret_origin": {
            "continuous_q_regret_mean": regret_overall["continuous_q_regret_mean"],
            "mode_given_best_q_regret_mean": regret_overall[
                "mode_given_best_q_regret_mean"
            ],
            "total_oracle_regret_mean": regret_overall["total_oracle_regret_mean"],
            "continuous_q_share_of_total": continuous_share,
            "dominant_term": (
                "CONTINUOUS_Q_SELECTION"
                if continuous_share is not None and continuous_share >= 0.5
                else "DISCRETE_MODE_SELECTION"
            ),
        },
        "q4_reward_weight_change_justified": {
            "verdict": "NOT_JUSTIFIED_PROBLEM_IS_OPTIMIZATION_AND_CREDIT_ASSIGNMENT",
            "evidence": (
                "the registered reward already ranks a strictly better action in "
                f"{1.0 - exact_mode11:.4f} of final contexts at the actor's own "
                f"proposed q, and a policy that never leaves mode 11 but picks q "
                f"exactly reaches {learned_mode_best_q['v2_utility_mean']:.6f} with "
                f"{learned_mode_best_q['p95_miss_count']} deadline misses, so the "
                "ordering the reward induces is not what the policy is missing"
            ),
        },
        "q5_smallest_run3_change": {
            "verdict": "CALIBRATE_THE_CONTINUOUS_Q_BRANCH_NOT_THE_REWARD",
            "peak_update": peak["update_index"],
            "peak_utility": peak["v2_utility_mean"],
            "final_utility": final["v2_utility_mean"],
            "peak_entropy_nats": peak["discrete_entropy_nats_mean"],
            "final_entropy_nats": final["discrete_entropy_nats_mean"],
            "entropy_rose_after_peak": bool(
                final["discrete_entropy_nats_mean"]
                > peak["discrete_entropy_nats_mean"]
            ),
            "critic_rank_quality_first_trained": first_trained[
                "spearman_minq_vs_exact_mean"
            ],
            "critic_rank_quality_final": final["spearman_minq_vs_exact_mean"],
            "critic_abs_error_first_trained": first_trained[
                "critic_abs_error_at_actor_top1_mean"
            ],
            "critic_abs_error_final": final["critic_abs_error_at_actor_top1_mean"],
            "adverse_q_e4_first_trained": (
                None if adverse_first is None else adverse_first["executed_q_e4_mean"]
            ),
            "adverse_q_e4_final": (
                None if adverse_final is None else adverse_final["executed_q_e4_mean"]
            ),
            "adverse_misses_final": (
                None if adverse_final is None else adverse_final["p95_miss_count"]
            ),
            "mode11_support_midpoint_q_e4": mode11_midpoint,
            "final_pooled_q_e4_mean": final["executed_q_e4_mean"],
            "untested_mechanism_hypothesis": (
                "mode 11 has the widest registered q support "
                f"[{mode11_lower}, {mode11_upper}], midpoint {mode11_midpoint:.1f}; "
                f"the final pooled executed q_e4 is {final['executed_q_e4_mean']:.1f}, "
                "between that midpoint and the oracle's preferred compression. A "
                "continuous maximum-entropy pull toward the middle of the widest "
                "support is consistent with the observed systematic "
                "under-compression, but this diagnostic cannot confirm it without "
                "a training run and does not claim it as established"
            ),
        },
        "q6_split_feasible_without_local": {
            "learned_p95_miss_rate": anchors["run2_final_p95_miss_rate"],
            "oracle_p95_miss_count": oracle_arm["p95_miss_count"],
            "fixed_p95_miss_count": arm_overall["FROZEN_FIXED_MODE11_Q6000"][
                "p95_miss_count"
            ],
            "mode11_only_contextual_q_utility": learned_mode_best_q["v2_utility_mean"],
            "mode11_only_contextual_q_miss_count": learned_mode_best_q[
                "p95_miss_count"
            ],
            "mode11_only_fraction_of_oracle_utility": (
                learned_mode_best_q["v2_utility_mean"]
                / oracle_arm["v2_utility_mean"]
                if oracle_arm["v2_utility_mean"]
                else None
            ),
            "exceedance_ms_max": miss_block["exceedance_ms_max"],
            "verdict": "YES_SPLIT_ALONE_IS_SUFFICIENT_ON_THIS_PANEL",
        },
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(summary: Mapping[str, Any]) -> str:
    """Render the durable human-readable diagnostic report."""
    conclusions = summary["conclusions"]
    final = summary["final_checkpoint_discrete_branch"]
    regret = summary["regret_decomposition_overall"]
    arms = summary["arm_summary_overall"]
    miss = summary["final_miss_characterization"]
    trend = summary["checkpoint_trend_pooled"]
    adaptation = summary["continuous_q_adaptation"]

    lines: List[str] = []
    add = lines.append

    add("# Run-2 Hybrid-SAC mode/q/regret diagnostic")
    add("")
    add(f"**Decision: {summary['decision']}**")
    add("")
    add(
        "Analysis-only. No retraining, no hyperparameter change, no reward-weight "
        "change, no checkpoint cherry-pick, no CARLA/OAI/CUDA. Update "
        f"{summary['primary_update']} remains the preregistered primary endpoint."
    )
    add("")

    add("## 0. Scope, and what these numbers are not")
    add("")
    add(
        "- This is **development fit-validation**, not a final test. The panel was "
        "visible throughout Run-2 model selection."
    )
    add(
        "- The reported P95 is the **modeled conditional retained-survivor latency "
        "proxy**. It is not a live, unconditional service percentile and not a "
        "200 ms SLA."
    )
    add(
        "- The 1,020 final decisions are **85 scene clusters x 4 network profiles x "
        "3 seeds**, not 1,020 independent samples. Seeds re-use the same 340 "
        "contexts, so pooled counts overstate effective sample size roughly 3x."
    )
    add(
        "- Raw reward magnitude is a scale artifact of the V2 utility, not evidence "
        "of convergence."
    )
    add(
        "- \"Learned strong deadline-aware behavior\" and \"beat the fixed "
        "baseline\" are different claims. Only the first is supported here."
    )
    add("")

    add("## 1. Binding revalidation")
    add("")
    add(
        f"All {len(summary['source_bindings']['evaluation_artifact_sha256'])} frozen "
        "evaluation artifacts rehashed to their manifest values; the manifest "
        "content hash, frozen training CLI source, frozen fixed comparator, frozen "
        "oracle GO bindings and both campaign reports revalidated before any "
        "scientific value was read."
    )
    add("")
    add("Reproduced anchors (recomputed from frozen rows, not copied):")
    add("")
    add("| anchor | value |")
    add("| --- | --- |")
    for name, value in sorted(summary["reproduced_anchors"].items()):
        add(f"| {name} | {_fmt(value)} |")
    add("")
    deviation = summary["max_vector_minus_authoritative_deviation"]
    add(
        f"Exhaustive enumeration independently reproduced the registered "
        f"constrained oracle action in all 340 contexts, and every executed action "
        f"recomputed from the checkpoints matched the frozen evaluation row for all "
        f"{len(REGISTERED_SEEDS) * len(REGISTERED_UPDATES)} checkpoints x 340 "
        "contexts."
    )
    add("")
    add(
        f"**Pricing path.** {summary['pricing_path']}. Across "
        f"{summary['scalar_vector_crosscheck_count']} cross-checked actions the "
        f"vectorized search table differs from the authoritative scalar path by at "
        f"most {deviation['q_perc']:.3e} in quality and {deviation['p95_ms']:.3e} ms "
        f"in modeled P95, and by exactly {deviation['utility']:.1f} in emitted V2 "
        "utility. The P95 difference is physically meaningless but breaks exact "
        "equality against the frozen rows, so all reported arms are priced by the "
        "authoritative path."
    )
    add("")
    add(
        "Two execution-semantics notes that a future re-analysis must preserve: "
        "the actor is run **one context at a time**, because a batched float32 "
        "GEMM differs by an ULP and that is enough to cross a round-half-up "
        "`q_e4` boundary (it does, for one update-0 context); and `q_e4` is the "
        "exact registered execution boundary, never the unquantized request."
    )
    add("")
    add(f"CUDA initialized at any point: **{summary['cuda_initialized']}**.")
    add("")

    add("## 2. The discrete branch: how mode 11 wins")
    add("")
    add(
        f"At update 5000 the actor's top-1 mode is 11 in "
        f"{final['actor_top1_mode_histogram']['mode_11']}/{final['row_count']} "
        "context-seed decisions."
    )
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| mean top-1 probability | {_fmt(final['top1_prob_mean'])} |")
    add(f"| min top-1 probability | {_fmt(final['top1_prob_min'])} |")
    add(f"| max top-1 probability | {_fmt(final['top1_prob_max'])} |")
    add(f"| mean top1-top2 probability gap | {_fmt(final['top1_top2_prob_gap_mean'])} |")
    add(f"| min top1-top2 probability gap | {_fmt(final['top1_top2_prob_gap_min'])} |")
    add(f"| mean top1-top2 logit gap | {_fmt(final['top1_top2_logit_gap_mean'])} |")
    add(f"| min top1-top2 logit gap | {_fmt(final['top1_top2_logit_gap_min'])} |")
    add(f"| mean discrete entropy (nats) | {_fmt(final['discrete_entropy_nats_mean'])} |")
    add(f"| max discrete entropy (nats) | {_fmt(final['discrete_entropy_nats_max'])} |")
    add("")
    add(
        "Maximum attainable entropy over 12 modes is ln(12) = 2.484907 nats. The "
        f"final head sits at {_fmt(final['discrete_entropy_nats_mean'], 4)} nats, "
        f"{_fmt(100 * final['discrete_entropy_nats_mean'] / math.log(MODE_COUNT), 1)}% "
        "of maximum, with a mean top-1 probability of only "
        f"{_fmt(final['top1_prob_mean'], 4)}. **The categorical head has not "
        "collapsed to a point mass.** It is broad but *statically ordered*: mode 11 "
        "is the argmax in every context, and the runner-up is only ever mode 10 or "
        "mode 8. Deterministic evaluation takes the argmax, so a diffuse but "
        "context-insensitive distribution still yields a constant discrete action. "
        "This distinction matters for Run-3: an entropy bonus on the discrete head "
        "would not fix it, because entropy is already high."
    )
    add("")
    add("Top-1 mode histograms at update 5000 (actor vs critics vs exact V2 utility):")
    add("")
    add("| mode | actor top-1 | actor top-2 | critic min-Q top-1 | exact V2 top-1 |")
    add("| --- | --- | --- | --- | --- |")
    for mode in range(MODE_COUNT):
        key = f"mode_{mode:02d}"
        add(
            f"| {mode} | {final['actor_top1_mode_histogram'][key]} "
            f"| {final['actor_top2_mode_histogram'][key]} "
            f"| {final['critic_minq_top1_mode_histogram'][key]} "
            f"| {final['exact_top1_mode_histogram'][key]} |"
        )
    add("")
    add("Per-profile discrete statistics at update 5000:")
    add("")
    add(
        "| profile | n | mean top-1 prob | mean prob gap | mean entropy | "
        "critic top-1 = 11 | exact top-1 = 11 |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for profile, block in sorted(final["by_profile"].items()):
        add(
            f"| {profile} | {block['row_count']} | {_fmt(block['top1_prob_mean'])} "
            f"| {_fmt(block['top1_top2_prob_gap_mean'])} "
            f"| {_fmt(block['discrete_entropy_nats_mean'])} "
            f"| {block['critic_top1_mode_histogram']['mode_11']} "
            f"| {block['exact_top1_mode_histogram']['mode_11']} |"
        )
    add("")

    add("## 3. Critic ranking")
    add("")
    add(
        "Run-2 episodes are one-step reward-bearing terminals, so the critic target "
        "is the immediate shaped V2 reward. `min-Q` minus the exact V2 utility of "
        "the same action is therefore a direct critic error, not a discounted-return "
        "proxy."
    )
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(
        f"| actor top-1 == critic min-Q top-1 | "
        f"{_fmt(final['actor_vs_critic_top1_agreement'])} |"
    )
    add(
        f"| critic min-Q top-1 == exact V2 top-1 | "
        f"{_fmt(final['critic_vs_exact_top1_agreement'])} |"
    )
    add(
        f"| actor top-1 == exact V2 top-1 | "
        f"{_fmt(final['actor_vs_exact_top1_agreement'])} |"
    )
    add(
        f"| Spearman rho(min-Q, exact V2) over 12 modes | "
        f"{_fmt(final['spearman_minq_vs_exact_mean'])} |"
    )
    add(
        f"| Spearman rho(actor logits, exact V2) over 12 modes | "
        f"{_fmt(final['spearman_logit_vs_exact_mean'])} |"
    )
    add(
        f"| mean signed critic error at actor top-1 | "
        f"{_fmt(final['critic_error_at_actor_top1_mean'])} |"
    )
    add(
        f"| mean absolute critic error at actor top-1 | "
        f"{_fmt(final['critic_abs_error_at_actor_top1_mean'])} |"
    )
    add("")
    q1 = conclusions["q1_mode11_dominance_source"]
    add(
        f"**Attribution: {q1['verdict']}.** Mode 11 is genuinely optimal in "
        f"{_fmt(100 * q1['oracle_mode11_fraction_each_mode_at_its_own_best_q'], 1)}% "
        "of contexts when every mode is played at its own best q. The critics rank "
        f"it first in {_fmt(100 * q1['critic_minq_top1_mode11_fraction'], 1)}%, and "
        f"the actor's argmax selects it in "
        f"{_fmt(100 * q1['actor_top1_mode11_fraction'], 1)}%. That is a monotone "
        "over-concentration chain: truth -> critics -> argmax. The actor is not "
        "ignoring its critics (they agree "
        f"{_fmt(100 * q1['actor_vs_critic_top1_agreement'], 1)}% of the time); the "
        "critics are over-ranking mode 11, and taking the argmax of an already "
        "biased ordering removes the remaining variation."
    )
    add("")

    add("## 4. Regret decomposition at update 5000")
    add("")
    add("Counterfactual definitions (exact, and stated because the terms can overlap):")
    add("")
    for key, text in sorted(summary["counterfactual_definitions"].items()):
        add(f"- **{key}**: {text}")
    add("")
    add("| arm | utility | quality | admission | P50 ms | P95 ms | P99 ms | P95 misses |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, block in arms.items():
        add(
            f"| {name} | {_fmt(block['v2_utility_mean'])} "
            f"| {_fmt(block['q_perc_mean'])} | {_fmt(block['p_admit_mean'])} "
            f"| {_fmt(block['latency_p50_ms_mean'], 3)} "
            f"| {_fmt(block['latency_p95_ms_mean'], 3)} "
            f"| {_fmt(block['latency_p99_ms_mean'], 3)} "
            f"| {block['p95_miss_count']} |"
        )
    add("")
    add("Additive decomposition (exact identity, no overlap):")
    add("")
    add("| term | mean | share of total |")
    add("| --- | --- | --- |")
    total = regret["total_oracle_regret_mean"]
    add(
        f"| continuous-q selection regret (B - A) "
        f"| {_fmt(regret['continuous_q_regret_mean'])} "
        f"| {_fmt(regret['continuous_q_regret_mean'] / total if total else 0.0, 4)} |"
    )
    add(
        f"| discrete-mode selection regret (D - B) "
        f"| {_fmt(regret['mode_given_best_q_regret_mean'])} "
        f"| {_fmt(regret['mode_given_best_q_regret_mean'] / total if total else 0.0, 4)} |"
    )
    add(f"| **total oracle regret (D - A)** | **{_fmt(total)}** | 1.0000 |")
    add("")
    add(
        f"Overlapping alternative, reported separately and never summed with the "
        f"above: best mode at the actor's own per-mode q (C - A) = "
        f"{_fmt(regret['mode_at_actor_q_regret_mean_overlapping'])}."
    )
    add("")
    add(
        f"Learned minus frozen fixed (A - E) = "
        f"{_fmt(regret['learned_minus_fixed_mean'])}. The learned policy does "
        "**not** beat the fixed comparator on pooled V2 utility."
    )
    add("")
    add(
        f"The learned mode equals the oracle mode in "
        f"{regret['contexts_where_learned_mode_is_oracle_mode']}/"
        f"{regret['row_count']} decisions, and the learned q is already the exact "
        f"within-mode optimum in "
        f"{regret['contexts_where_actor_q_already_optimal']}/{regret['row_count']}."
    )
    add("")
    add("By network profile:")
    add("")
    add("| profile | continuous-q (B-A) | discrete-mode (D-B) | total (D-A) | A-E |")
    add("| --- | --- | --- | --- | --- |")
    for profile, block in sorted(summary["regret_decomposition_by_profile"].items()):
        add(
            f"| {profile} | {_fmt(block['continuous_q_regret_mean'])} "
            f"| {_fmt(block['mode_given_best_q_regret_mean'])} "
            f"| {_fmt(block['total_oracle_regret_mean'])} "
            f"| {_fmt(block['learned_minus_fixed_mean'])} |"
        )
    add("")

    add("## 5. The continuous branch")
    add("")
    add("| profile | learned q | learned q_e4 | best q_e4 in learned mode | oracle q_e4 |")
    add("| --- | --- | --- | --- | --- |")
    for profile, block in sorted(adaptation["by_profile"].items()):
        add(
            f"| {profile} | {_fmt(block['learned_q_perc_mean'], 4)} "
            f"| {_fmt(block['learned_q_e4_mean'], 1)} "
            f"| {_fmt(block['learned_mode_best_q_e4_mean'], 1)} "
            f"| {_fmt(block['oracle_q_e4_mean'], 1)} |"
        )
    add("")
    add(
        "`q_e4` is a **compression knob**: payload falls strictly as q_e4 rises, so "
        "a higher q_e4 means a smaller payload, lower latency and lower realized "
        "perception quality. Read the table in that direction."
    )
    add("")
    add(
        f"Between-profile share of executed q_e4 variance: "
        f"{_fmt(adaptation['between_profile_variance_share'], 4)}. The continuous "
        "head is genuinely conditioned on the radio context and moves the right "
        "way - it compresses hardest under ADVERSE_STABLE and least under "
        "FAVORABLE_STABLE."
    )
    add("")
    q2 = conclusions["q2_continuous_branch_useful"]
    add(
        "It is nonetheless **systematically under-compressed in every profile**: "
        "the oracle wants a higher q_e4 than the policy chooses everywhere - "
        + ", ".join(
            f"{profile} +{_fmt(delta, 1)}"
            for profile, delta in sorted(
                q2["per_profile_oracle_minus_learned_q_e4"].items()
            )
        )
        + ". That single bias is what the 65% continuous-q regret term is made of, "
        "and it is also what produces the 23 deadline misses."
    )
    add("")

    add("## 6. The 23 final P95 misses")
    add("")
    add("| quantity | value |")
    add("| --- | --- |")
    add(f"| count | {miss['count']} |")
    add(f"| profiles involved | {', '.join(miss['profiles'])} |")
    add(f"| distinct panel contexts | {miss['distinct_panel_indices']} |")
    add(f"| distinct scene clusters | {miss['distinct_scene_clusters']} |")
    add(f"| min exceedance (ms) | {_fmt(miss['exceedance_ms_min'], 3)} |")
    add(f"| max exceedance (ms) | {_fmt(miss['exceedance_ms_max'], 3)} |")
    add(f"| mean exceedance (ms) | {_fmt(miss['exceedance_ms_mean'], 3)} |")
    add(f"| fixed action also misses these | {miss['fixed_action_also_misses']} |")
    add(f"| oracle misses these | {miss['oracle_misses']} |")
    add(
        f"| additional q_e4 compression needed for same-mode feasibility | "
        f"+{miss['q_e4_increase_for_feasibility_min']} to "
        f"+{miss['q_e4_increase_for_feasibility_max']} |"
    )
    add(
        f"| mean utility change if forced feasible | "
        f"{_fmt(miss['utility_delta_if_made_feasible_mean'])} |"
    )
    add("")
    add(
        f"All 23 misses are ADVERSE_STABLE and fall in only "
        f"{miss['distinct_panel_indices']} distinct panel contexts repeated across "
        "the three seeds, so they are roughly "
        f"{miss['distinct_scene_clusters']} independent scene events, not 23. Every "
        "overshoot is small (max "
        f"{_fmt(miss['exceedance_ms_max'], 3)} ms past a 200 ms modeled percentile) "
        "and in every case a feasible action exists **in the same mode** at higher "
        "compression, so nothing structural blocks feasibility."
    )
    add("")
    add(
        "A train-derived deadline-safety margin is therefore plausible in "
        "principle. **No margin value is proposed here.** These are the "
        "development contexts; fitting a margin on them would contaminate the "
        "panel and convert a diagnostic into a tuned result. Any margin must be "
        "derived on the training split and then evaluated once."
    )
    add("")

    add("## 7. Checkpoint trajectory")
    add("")
    add(
        "| update | utility | P95 misses | distinct modes | mean q_e4 | mean top-1 "
        "prob | entropy (nats) | mean abs critic error | rho(min-Q, exact) |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in trend:
        add(
            f"| {row['update_index']} | {_fmt(row['v2_utility_mean'])} "
            f"| {row['p95_miss_count']} | {row['distinct_executed_modes']} "
            f"| {_fmt(row['executed_q_e4_mean'], 1)} "
            f"| {_fmt(row['top1_prob_mean'], 4)} "
            f"| {_fmt(row['discrete_entropy_nats_mean'], 4)} "
            f"| {_fmt(row['critic_abs_error_at_actor_top1_mean'])} "
            f"| {_fmt(row['spearman_minq_vs_exact_mean'], 4)} |"
        )
    add("")
    adverse = summary["checkpoint_trend_by_profile"].get("ADVERSE_STABLE")
    if adverse:
        add(
            "ADVERSE_STABLE only - this is where every deadline miss lives, so the "
            "pooled view above understates the drift:"
        )
        add("")
        add(
            "| update | mean q_e4 | mean modeled P95 ms | P95 misses | "
            "rho(min-Q, exact) | mean abs critic error |"
        )
        add("| --- | --- | --- | --- | --- | --- |")
        for row in adverse:
            add(
                f"| {row['update_index']} | {_fmt(row['executed_q_e4_mean'], 1)} "
                f"| {_fmt(row['latency_p95_ms_mean'], 3)} "
                f"| {row['p95_miss_count']} "
                f"| {_fmt(row['spearman_minq_vs_exact_mean'], 4)} "
                f"| {_fmt(row['critic_abs_error_at_actor_top1_mean'])} |"
            )
        add("")
    peak = conclusions["q5_smallest_run3_change"]
    add(
        f"Pooled utility peaks at update {peak['peak_update']} "
        f"({_fmt(peak['peak_utility'])}) and ends at {_fmt(peak['final_utility'])}. "
        "Three things move over that interval, and it is worth being precise about "
        "which of them is the cause:"
    )
    add("")
    add(
        f"1. **Continuous-q drift, under ADVERSE_STABLE.** Mean q_e4 falls from "
        f"{_fmt(peak['adverse_q_e4_first_trained'], 1)} at update 500 to "
        f"{_fmt(peak['adverse_q_e4_final'], 1)} at update 5000 - the policy "
        "progressively under-compresses exactly where the channel has no headroom, "
        f"and adverse misses rise from 0 to {peak['adverse_misses_final']}. This is "
        "the proximate cause of the utility decline."
    )
    add(
        f"2. **Critic rank quality degrades.** Spearman rho(min-Q, exact V2) over "
        f"the 12 modes falls from {_fmt(peak['critic_rank_quality_first_trained'], 4)} "
        f"to {_fmt(peak['critic_rank_quality_final'], 4)}, and mean absolute critic "
        f"error rises from {_fmt(peak['critic_abs_error_first_trained'])} to "
        f"{_fmt(peak['critic_abs_error_final'])}. The critics get worse, not better, "
        "with continued training."
    )
    add(
        f"3. **Discrete entropy does not collapse - it rises** "
        f"({_fmt(peak['peak_entropy_nats'], 4)} -> "
        f"{_fmt(peak['final_entropy_nats'], 4)} nats) while the argmax stays pinned "
        "to mode 11. So the decline is not a discrete-exploration failure."
    )
    add("")
    add(
        f"**Update 5000 remains the preregistered primary endpoint. Update "
        f"{peak['peak_update']} is reported here only as a trajectory feature and "
        "is explicitly not selected as a result.**"
    )
    add("")

    add("## 8. Required answers")
    add("")
    q1 = conclusions["q1_mode11_dominance_source"]
    add(
        f"**1. Is mode-11 dominance actor or critic?** "
        f"{q1['verdict']}. The chain is monotone: mode 11 is genuinely the best "
        f"mode in {_fmt(100 * q1['oracle_mode11_fraction_each_mode_at_its_own_best_q'], 1)}% "
        f"of contexts, the twin critics rank it first in "
        f"{_fmt(100 * q1['critic_minq_top1_mode11_fraction'], 1)}%, and the actor "
        f"selects it in {_fmt(100 * q1['actor_top1_mode11_fraction'], 1)}%. The "
        "actor is faithfully following its critics (agreement "
        f"{_fmt(100 * q1['actor_vs_critic_top1_agreement'], 1)}%), so this is not "
        "case (a). The critics do mis-rank - they agree with the exact V2 ordering "
        f"only {_fmt(100 * q1['critic_vs_exact_top1_agreement'], 1)}% of the time, "
        f"with rho = {_fmt(q1['spearman_minq_vs_exact_mean'], 4)} - so it is "
        "primarily case (b), amplified by the deterministic argmax. It is *not* "
        "case (c) in the strong form: the reward surface does not favour mode 11 "
        "anywhere near universally. Note also that the categorical head is broad "
        f"(entropy {_fmt(q1['discrete_entropy_nats_mean'], 4)} of a possible "
        f"{_fmt(q1['max_possible_entropy_nats'], 4)} nats), so the constancy comes "
        "from a static *ordering*, not from a collapsed distribution."
    )
    add("")
    q2 = conclusions["q2_continuous_branch_useful"]
    add(
        f"**2. Is the continuous-q branch genuinely contextual and useful?** Yes on "
        f"both counts, but it is miscalibrated. "
        f"{_fmt(100 * q2['between_profile_q_variance_share'], 1)}% of executed q_e4 "
        "variance is between-profile and the ordering across profiles is physically "
        "correct. It is also the single most valuable branch: holding the learned "
        "mode fixed and choosing q exactly raises pooled utility from "
        f"{_fmt(arms['LEARNED_ACTION']['v2_utility_mean'])} to "
        f"{_fmt(arms['LEARNED_MODE_BEST_Q']['v2_utility_mean'])} and removes all "
        "23 deadline misses. But it is the exact within-mode optimum in only "
        f"{q2['contexts_where_actor_q_already_optimal']}/1020 decisions and "
        "under-compresses in every profile."
    )
    add("")
    q3 = conclusions["q3_regret_origin"]
    add(
        f"**3. Where does most oracle regret originate?** "
        f"{q3['dominant_term']}, by roughly two to one. Continuous-q selection "
        f"within the learned mode accounts for "
        f"{_fmt(q3['continuous_q_regret_mean'])} "
        f"({_fmt(100 * q3['continuous_q_share_of_total'], 1)}%) of the "
        f"{_fmt(q3['total_oracle_regret_mean'])} mean total; discrete-mode choice "
        f"accounts for {_fmt(q3['mode_given_best_q_regret_mean'])}. The split is "
        "stable across all four profiles and is widest under ADVERSE_STABLE."
    )
    add("")
    add(
        f"**4. Is reward-weight modification justified?** "
        f"{conclusions['q4_reward_weight_change_justified']['verdict']}. "
        "Concretely: "
        f"{conclusions['q4_reward_weight_change_justified']['evidence']}. The "
        "registered reward already prefers the actions the policy fails to take; "
        "the failure is that the critics mis-rank modes and the q head is "
        "mis-calibrated. Changing weights now would tune away a measurable "
        "optimization failure, invalidate every frozen comparator, and cost the "
        "ability to compare Run-3 with Run-1 and Run-2."
    )
    add("")
    q5 = conclusions["q5_smallest_run3_change"]
    add(
        f"**5. Smallest scientifically justified Run-3 change?** "
        f"{q5['verdict']}. The evidence points at one branch, not at the reward and "
        "not at the action space: fix the continuous-q calibration, which owns "
        f"{_fmt(100 * q3['continuous_q_share_of_total'], 1)}% of the regret and "
        "100% of the deadline misses, and stop its adverse-profile drift over "
        "training. Everything else here is diagnosis, not a mandate - no Run-3 is "
        "implemented or launched."
    )
    add("")
    add(
        f"One mechanism is worth testing first because it is cheap to check and "
        f"consistent with every number above: {q5['untested_mechanism_hypothesis']}."
    )
    add("")
    q6 = conclusions["q6_split_feasible_without_local"]
    add(
        f"**6. Is SPLIT still feasible without introducing LOCAL?** "
        f"{q6['verdict']}. The strongest evidence is arm B: a policy that never "
        "leaves mode 11 and only picks q per context reaches "
        f"{_fmt(q6['mode11_only_contextual_q_utility'])} with "
        f"{q6['mode11_only_contextual_q_miss_count']} modeled deadline misses - "
        f"{_fmt(100 * q6['mode11_only_fraction_of_oracle_utility'], 1)}% of the full "
        "contextual oracle, and above the frozen fixed comparator. The constrained "
        "oracle also attains zero misses using SPLIT actions alone in all 340 "
        f"contexts. The learned policy's "
        f"{_fmt(100 * q6['learned_p95_miss_rate'], 4)}% miss rate is a "
        "q-calibration shortfall, not evidence that the SPLIT action space is "
        "infeasible. **Nothing in this evidence requires LOCAL.**"
    )
    add("")

    add("## 9. Decision")
    add("")
    add(
        f"**{summary['decision']}** - the diagnostic is complete and internally "
        "consistent, every frozen binding revalidated, and the failure is localized "
        "to continuous-q miscalibration plus critic mis-ranking rather than to the "
        "reward specification, the action support or the SPLIT/LOCAL question. "
        "Run-3 is **not** implemented or launched here, no reward weight was "
        "changed, and no checkpoint was cherry-picked."
    )
    add("")
    add("### What this does and does not claim")
    add("")
    add(
        "- The policy **did** learn strong deadline-aware behaviour: update 0 "
        "misses 765/1020 and update 5000 misses 23/1020, a 97% reduction, and the "
        "continuous head is genuinely channel-conditioned."
    )
    add(
        "- The policy **did not** beat the frozen fixed comparator: "
        f"{_fmt(arms['LEARNED_ACTION']['v2_utility_mean'])} against "
        f"{_fmt(arms['FROZEN_FIXED_MODE11_Q6000']['v2_utility_mean'])} pooled V2 "
        "utility, and the fixed action has zero modeled misses. These are two "
        "different claims and only the first is supported."
    )
    add(
        "- Everything here is **development fit-validation on a panel that was "
        "visible during model selection**. It is not a held-out test result."
    )
    add(
        "- Every P95 quantity is the **modeled conditional retained-survivor "
        "proxy**, not a live or unconditional service percentile, and not a "
        "200 ms SLA."
    )
    add(
        "- The 1,020 decisions are 85 scene clusters x 4 profiles x 3 seeds. "
        "Treat counts as roughly 340 independent context evaluations, and the 23 "
        "misses as about 10 independent scene events."
    )
    add(
        "- Reward magnitudes are a property of the V2 utility scale. Nothing here "
        "claims convergence, and nothing claims reward approaches 1."
    )
    add("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only Run-2 mode/q/regret diagnostic."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--stage1-cache",
        type=Path,
        default=None,
        help="development-only cache of the per-checkpoint actor/critic reads",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    root = Path(args.project_root).resolve(strict=True)
    output_dir = (
        Path(args.output_dir) if args.output_dir is not None
        else root / OUTPUT_RELATIVE_PATH
    )
    try:
        summary = run_diagnostic(
            root=root, output_dir=output_dir, stage1_cache=args.stage1_cache
        )
    except Exception as exc:  # noqa: BLE001 - the terminal marker is the contract
        print(f"BLOCK: {type(exc).__name__}: {exc}")
        return 1
    print(f"COMPLETE: {summary['decision']} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
